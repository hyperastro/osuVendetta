from sklearn.metrics import f1_score
import torch.nn as nn
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import optuna
from optuna.trial import Trial
import torch.optim as optim
import multiprocessing
import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import requests
import time
import zipfile



# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = 'SmallerParsedReplays'
csv_file = 'newcsvfile2small.csv'
study_name = "64x2FirstRun"
server_adr = "http://188.251.214.6:7270"
DATABASE_URL = "postgresql+psycopg2://osuvendetta:osuvendetta@188.251.214.6:5432/optuna_study_db"

# Dataset setup flag
dataset_initialized = False
study_initialized = False
init_done_file = os.path.join(data_dir, '.init_done')

def DetermineBatchSize():
    freemem = torch.cuda.mem_get_info()[0] / 1024 ** 3
    if freemem < 8: #less than 8Gb of Vram
        return 128
    elif freemem < 12:#less than 12Gb of Vram
        return 256
    elif freemem < 17: #less than 17Gb of Vram
        return 512
    else:
        return 1024 #anything else (this won't be used unless u have a rtx 4090 or something)



def Normal(): #full power mode
    global num_workers
    global batchsize
    num_workers = num_workers
    batchsize = batchsize
    return num_workers,batchsize

def Halfpower(): #half power ....
    global num_workers
    global batchsize
    num_workers = round(num_workers/2)
    batchsize = batchsize/2
    return num_workers,batchsize

def Lowpowermode(): #low power might remove it later its kinda unproductive because of the low batch size
    global num_workers
    global batchsize
    num_workers = round(num_workers/4)
    batchsize = round(batchsize/4)
    return num_workers,batchsize


def customsettings(): #custom setting ik its a bit spaghetti sorry to who ever tries to read it
    global num_workers
    global batchsize
    print(f"recommended a batchsize of {batchsize}, and a thread count of {num_workers}")
    batchsize = int(input("Insert custom batchsize: "))
    num_workers = int(input("Insert custom thread count: "))
    if batchsize > DetermineBatchSize():
        print(f"Warning custom batch size of {batchsize} is bigger than recommended batch size of {DetermineBatchSize()}")
    currentcpucount = multiprocessing.cpu_count()
    if num_workers > currentcpucount:
        print(f"Warining custom thread count {num_workers} exceeds cpu core count {currentcpucount}")
    return num_workers,batchsize









def download_chunks(data_dir, server_adr): #download chunks func as you can guess it download missing chunks for the dataset
    try:

        os.makedirs(data_dir, exist_ok=True)
        with open('downloaded_chunks', 'r') as chunkline:
            chunknumber = int(chunkline.readline().strip())

        if chunknumber >= 64:
            print("All chunks downloaded, skipping step...")
            return

        while chunknumber < 64:
            try:

                download_url = f"{server_adr}/download_chunk?chunk_number={chunknumber}"
                print(f"Downloading chunk_{chunknumber} ")

                response = requests.get(download_url, timeout=60)
                response.raise_for_status()

                zip_file_path = os.path.join(data_dir, f"chunk_{chunknumber}.zip")
                with open(zip_file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded chunk {chunknumber}")

                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    print(f"Decompressed chunk {chunknumber}")

                    os.remove(zip_file_path)
                    chunknumber += 1

                    with open('downloaded_chunks', 'w') as chunkline:
                        chunkline.write(str(chunknumber))

                except zipfile.BadZipFile:
                    print(f"Corrupted zip file for chunk {chunknumber}. Removing and retrying.")
                    if os.path.exists(zip_file_path):
                        os.remove(zip_file_path)
                    continue

            except requests.exceptions.RequestException as e:
                print(f"Failed to download chunk {chunknumber}: {e}")

                time.sleep(2)
                continue

    except FileNotFoundError:
        print("No download tracking file found. Creating new one...")
        with open("downloaded_chunks", 'w') as file:
            file.write('0')
        download_chunks(data_dir, server_adr)





class OsuReplayDataset(Dataset):
    def __init__(self, csv_file, data_dir, segment_size=1000, overlap_size=500, is_test=False,
                 log_file='bad_files_log.txt'):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.overlap_size = overlap_size
        self.is_test = is_test
        self.log_file = log_file
        self.data_frame = pd.read_csv(csv_file)

        # Ensure the necessary columns are in the CSV file
        required_columns = ['filename', 'class_normal', 'class_relax', 'class_frametime']
        if not all(col in self.data_frame.columns for col in required_columns):
            raise ValueError(f"CSV file must contain {required_columns} columns.")

        self.files = self.data_frame['filename'].values
        self.labels = self.data_frame[['class_normal', 'class_relax', 'class_frametime']].values.astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]
        file_path = os.path.join(self.data_dir, f"{os.path.splitext(file)[0]}.npy")

        try:
            # Load preprocessed data from the .npy file
            data = np.load(file_path)

            # Segmenting the data
            segments = []
            for start in range(0, len(data) - self.segment_size + 1, self.segment_size - self.overlap_size):
                end = start + self.segment_size
                segment = data[start:end]
                if len(segment) < self.segment_size:
                    padding = np.zeros((self.segment_size - len(segment), segment.shape[1]))
                    segment = np.vstack((segment, padding))
                segments.append(segment)

            if not segments:
                raise ValueError(f"No valid segments in file {file}")

            # Select a random segment
            segment = torch.tensor(segments[np.random.randint(len(segments))], dtype=torch.float32)
            if torch.isnan(segment).any() or torch.isinf(segment).any():
                raise ValueError(f"NaN or infinite values found in file {file}")
        except Exception as e:
            # Log any file processing issues
            with open(self.log_file, 'a') as log_f:
                log_f.write(f"{file}\n")
            print(f"Error processing file {file}: {e}")
            segment = torch.zeros((self.segment_size, 6), dtype=torch.float32)

        return segment, torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack([label.clone().detach() for label in labels])
    return sequences_padded, labels


def stratified_split(dataset, test_size=0.2):
    flattened_labels = [tuple(label) for label in dataset.labels]
    df = pd.DataFrame({'files': dataset.files, 'labels': flattened_labels})
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=flattened_labels)

    train_dataset = Subset(dataset, train_df.index)
    test_dataset = Subset(dataset, test_df.index)
    return train_dataset, test_dataset


def initialize_dataset():
    global dataset_initialized
    if not dataset_initialized:
        if os.path.exists(init_done_file):
            print("Dataset already initialized. Skipping initialization.")
            dataset = OsuReplayDataset(csv_file, data_dir)
            dataset_initialized = True
            return dataset

        dataset = OsuReplayDataset(csv_file, data_dir)
        dataset_initialized = True
        with open(init_done_file, 'w') as f:
            f.write('Initialization done')
        print("Dataset initialized successfully.")
        return dataset
    else:
        print("Dataset already initialized.")
        return OsuReplayDataset(csv_file, data_dir)


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.50):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # LSTM layer
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)  # Fully connected layer
        return torch.sigmoid(out)


input_size = 6
hidden_size = 64
output_size = 3
num_layers = 2
model = BiLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

class_counts = [10474, 10158, 10456]
total_samples = sum(class_counts)
class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
class_weights = torch.tensor(class_weights).to(device)

num_epochs = 30
max_grad_norm = 1.0
accumulation_steps = 2
train_losses = []
test_losses = []
best_f1 = 0.0
best_model_state = None
best_epoch = 0
f1_scores = []

def initialize_study():
    global study_initialized
    if not study_initialized:
        study = optuna.create_study(
            study_name=study_name,
            storage=DATABASE_URL,
            direction='maximize',
            load_if_exists=True
        )
        study_initialized = True
        return study
    else:
        print(f"Using an existing study with name '{study_name}'. Skipping new initialization.")
        return optuna.load_study(study_name=study_name, storage=DATABASE_URL)


def determine_batch_size(model, max_vram_usage=0.9, min_batch_size=1): #better function to determine batch size dynamicly 
    
    # Available VRAM in bytes
    available_vram = torch.cuda.mem_get_info()[0]
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    available_vram *= max_vram_usage  # Limit VRAM usage to a fraction for safety
    batch_size = available_vram // model_memory  # Estimate batch size based on available memory
    return max(min_batch_size, int(batch_size))  


def initialize_model(hidden_size, dropout=0.2):
    """Initializes the model based on parameters for memory estimation."""
    input_size, output_size, num_layers = 6, 3, 2
    model = BiLSTMModel(input_size, hidden_size, output_size, num_layers, dropout=dropout).to(device)
    return model


def performancesettings():
    global num_workers, batchsize
    validinput = False
    while not validinput:
        print(
            f"Choose performance settings(1-4):\n (1)Normal Perfomance(full computer utilization) \n (2)Half-power \n (3)Low-power \n (4)Custom settings ")
        useroption = int(input(f"choose an option (1-4): "))
        if useroption == 1:
            print(f"Normal performance chosen")
            validinput = True
        elif useroption == 2:
            print(f"Half-power chosen")
            validinput = True
            Halfpower()
        elif useroption == 3:
            print(f"Low-power chosen")
            validinput = True
            Lowpowermode()
        elif useroption == 4:
            validinput = True
            customsettings()
        else:
            print(f"Invalid input, please choose a number between 1 and 4")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_workers = multiprocessing.cpu_count()

    # Dataset loading
    dataset = initialize_dataset()
    if dataset is not None:
        train_dataset, test_dataset = stratified_split(dataset)


    def objective(trial: optuna.Trial):
        global batchsize  

        # Sample hyperparameters
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
        dropout = trial.suggest_float('dropout', 0.2, 0.5)
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.9)
        scheduler_patience = trial.suggest_int('scheduler_patience', 3, 10)
        scheduler_min_lr = trial.suggest_float('scheduler_min_lr', 1e-7, 1e-5, log=True)
        hidden_size = trial.suggest_int('hidden_size', 64, 128)

        model = initialize_model(hidden_size, dropout)

        batchsize = determine_batch_size(model)
        batchsize = round(batchsize/28)
        print(f"Calculated batch size: {batchsize}")
        performancesettings()

        train_dataloader = DataLoader(
            train_dataset, batch_size=batchsize, shuffle=True,
            collate_fn=collate_fn, num_workers=round(num_workers / 2),
            persistent_workers=True, pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batchsize, shuffle=False,
            collate_fn=collate_fn, num_workers=round(num_workers / 2),
            persistent_workers=True, pin_memory=True
        )

        # Training loop setup
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=scheduler_min_lr
        )

        best_f1 = 0.0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            # Training phase
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                      desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
                for i, (inputs, targets) in pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / accumulation_steps
                    loss.backward()

                    if (i + 1) % accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    running_loss += loss.item()
                    pbar.set_postfix({'loss': running_loss / (i + 1)})

            # Evaluation phase
            model.eval()
            test_loss = 0.0
            true_labels, pred_labels = [], []

            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
                    true_labels.extend(targets.cpu().numpy())
                    pred_labels.extend((outputs.cpu().numpy() > 0.5).astype(int))

            test_loss /= len(test_dataloader)
            f1 = f1_score(true_labels, pred_labels, average='micro')

            # Update best F1 score
            if f1 > best_f1:
                best_f1 = f1
            scheduler.step(f1)

        trial.report(best_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return best_f1


    # Main study setup
    study = initialize_study()
    study.optimize(objective, n_trials=1000, n_jobs=1)


    # Output best results
    print("Best trial:")
    trial = study.best_trial
    print(f"Best F1 Score: {trial.value}")
    print("Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
