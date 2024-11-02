import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import pickle
import optuna
from optuna.trial import Trial
import torch.optim as optim
import requests
from torch.utils.data import Dataset, DataLoader, Subset
import zipfile
import time
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device: {torch.cuda.current_device()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")


    def download_chunks(data_dir, server_adr):
        try:

            os.makedirs(data_dir, exist_ok=True)
            with open('downloaded_chunks', 'r') as chunkline:
                chunknumber = int(chunkline.readline().strip())

            if chunknumber >= 51:
                print("All chunks downloaded, skipping step...")
                return

            while chunknumber < 51:
                try:

                    download_url = f"{server_adr}/download_chunk?chunk_number={chunknumber + 1}"
                    print(f"Downloading chunk_{chunknumber + 1} ")

                    response = requests.get(download_url, timeout=60)
                    response.raise_for_status()

                    zip_file_path = os.path.join(data_dir, f"chunk_{chunknumber + 1}.zip")
                    with open(zip_file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded chunk {chunknumber + 1}")

                    try:
                        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                            zip_ref.extractall(data_dir)
                        print(f"Decompressed chunk {chunknumber + 1}")


                        os.remove(zip_file_path)
                        chunknumber += 1

                        with open('downloaded_chunks', 'w') as chunkline:
                            chunkline.write(str(chunknumber))

                    except zipfile.BadZipFile:
                        print(f"Corrupted zip file for chunk {chunknumber + 1}. Removing and retrying.")
                        if os.path.exists(zip_file_path):
                            os.remove(zip_file_path)
                        continue

                except requests.exceptions.RequestException as e:
                    print(f"Failed to download chunk {chunknumber + 1}: {e}")

                    time.sleep(2)
                    continue

        except FileNotFoundError:
            print("No download tracking file found. Creating new one...")
            with open("downloaded_chunks", 'w') as file:
                file.write('0')
            download_chunks(data_dir, server_adr)

    data_dir = 'SmallerParsedReplays'
    server_adr = "http://188.251.214.6:7270"
    download_chunks(data_dir, server_adr)





    class OsuReplayDataset(Dataset):
        def __init__(self, csv_file, data_dir, button_encoder=None, scaler=None, segment_size=1000, overlap_size=500,
                     is_test=False, log_file='bad_files_log.txt'):
            self.data_dir = data_dir
            self.segment_size = segment_size
            self.overlap_size = overlap_size
            self.is_test = is_test
            self.log_file = log_file

            self.data_frame = pd.read_csv(csv_file)

            if not all(col in self.data_frame.columns for col in
                       ['filename', 'class_normal', 'class_relax', 'class_frametime']):
                raise ValueError(
                    "CSV file must contain 'filename', 'class_normal', 'class_relax', 'class_frametime' columns.")

            self.files = self.data_frame['filename'].values
            self.labels = self.data_frame[['class_normal', 'class_relax', 'class_frametime']].values.astype(np.float32)


            if button_encoder is None:
                self.button_encoder = LabelEncoder()
                all_buttons = []
                for file in self.files:
                    try:
                        df = pd.read_csv(os.path.join(data_dir, file), header=None, usecols=[5], dtype=str)
                        all_buttons.extend(df[5].unique())
                    except Exception as e:
                        print(f"Error reading file {file}: {e}")
                self.button_encoder.fit(all_buttons)
                with open('button_encoder.pkl', 'wb') as f:
                    pickle.dump(self.button_encoder, f)
            else:
                self.button_encoder = button_encoder

            if scaler is None:
                self.scaler = StandardScaler()
                self._initialize_scaler(data_dir)
                with open('scaler_mean.pkl', 'wb') as f_mean, open('scaler_std.pkl', 'wb') as f_std:
                    pickle.dump(self.scaler.mean_, f_mean)
                    pickle.dump(self.scaler.scale_, f_std)
            else:
                self.scaler = scaler

        def _initialize_scaler(self, data_dir):
            sample_files = np.random.choice(self.files, min(1000, len(self.files)), replace=False)
            data_samples = []
            for file in sample_files:
                try:
                    df = pd.read_csv(os.path.join(data_dir, file), header=None, usecols=[0, 3, 4], dtype=float)
                    data_samples.append(df)
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
            data_samples = pd.concat(data_samples, axis=0)
            self.scaler.fit(data_samples)

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            file = self.files[idx]
            label = self.labels[idx]
            file_path = os.path.join(self.data_dir, file)

            if not os.path.exists(file_path):
                download_url = f"http://188.251.214.6:7270/download/{file}"
                response = requests.get(download_url)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            try:
                df = pd.read_csv(os.path.join(self.data_dir, file), header=None,
                                 dtype={0: float, 1: float, 2: float, 3: float, 4: float, 5: str})
                if df.empty or df.shape[1] != 6:  # Ensure the file is not empty and has exactly 6 columns
                    raise ValueError(f"File {file} is empty or malformed")

                df[5] = self.button_encoder.transform(df[5])
                df[[0, 3, 4]] = self.scaler.transform(df[[0, 3, 4]])

                segments = []
                for start in range(0, len(df) - self.segment_size + 1, self.segment_size - 500):
                    end = start + self.segment_size
                    segment = df.iloc[start:end].values

                    if len(segment) < self.segment_size:
                        padding = np.zeros((self.segment_size - len(segment), segment.shape[1]))
                        segment = np.vstack((segment, padding))
                    segments.append(segment)

                if not segments:
                    raise ValueError(f"No valid segments in file {file}")

                segment = torch.tensor(segments[np.random.randint(len(segments))], dtype=torch.float32)
                if torch.isnan(segment).any() or torch.isinf(segment).any():
                    raise ValueError(f"NaN or infinite values found in file {file}")
            except Exception as e:

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

        df = pd.DataFrame({
            'files': dataset.files,
            'labels': flattened_labels
        })

        train_df, test_df = train_test_split(df, test_size=test_size, stratify=flattened_labels)

        train_dataset = Subset(dataset, train_df.index)
        test_dataset = Subset(dataset, test_df.index)
        return train_dataset, test_dataset

    csv_file = 'newcsvfile2small.csv'
    data_dir = 'SmallerParsedReplays'

    if not os.listdir(data_dir):  # Check if directory is empty
        print("Data directory is empty. Downloading dataset...")
        download_chunks(data_dir, server_adr)

    # Load the dataset
    dataset = OsuReplayDataset(csv_file, data_dir)
    print(f"Loaded {len(dataset)} samples from {csv_file}")

    train_dataset, test_dataset = stratified_split(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

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
    print(f"Model is on device: {next(model.parameters()).device}")

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

    # Training loop


    DATABASE_URL = "postgresql+psycopg2://osuvendetta:osuvendetta@188.251.214.6:5432/optuna_study_db"


    # Distributed Optuna study setup
    study_name = "64x2FirstRun"
    study = optuna.create_study(
        study_name=study_name,
        storage=DATABASE_URL,
        direction='maximize',
        load_if_exists=True
    )

    def objective(trial: Trial):
        # Sample hyperparameters using Optuna's updated methods
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
        dropout = trial.suggest_float('dropout', 0.2, 0.5)
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.9)
        scheduler_patience = trial.suggest_int('scheduler_patience', 3, 10)
        scheduler_min_lr = trial.suggest_float('scheduler_min_lr', 1e-7, 1e-5, log=True)


        model = BiLSTMModel(input_size, hidden_size, output_size, num_layers, dropout=dropout).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=scheduler_min_lr
        )

        best_f1 = 0.0

        # Training loop with Optuna
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
                for i, (inputs, targets) in pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / accumulation_steps
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
            true_labels = []
            pred_labels = []

            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    true_labels.extend(targets.cpu().numpy())
                    pred_labels.extend((outputs.cpu().numpy() > 0.5).astype(int))

            test_loss /= len(test_dataloader)
            f1 = f1_score(true_labels, pred_labels, average='micro')

            # Update best F1 score
            if f1 > best_f1:
                best_f1 = f1

            # Step the scheduler
            scheduler.step(f1)

        # Report the best F1 score to Optuna
        trial.report(best_f1, epoch)


        if trial.should_prune():
            raise optuna.TrialPruned()

        return best_f1


    study.optimize(objective, n_trials=1000)  # Each machine can independently contribute to this

    print("Best trial:")
    trial = study.best_trial
    print(f"Best F1 Score: {trial.value}")
    print("Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
