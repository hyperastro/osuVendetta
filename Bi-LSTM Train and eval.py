import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device: {torch.cuda.current_device()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")


class OsuReplayDataset(Dataset):
    def __init__(self, data_dir, label_encoder=None, button_encoder=None, max_length=8000):
        self.data_dir = data_dir
        self.files = []
        self.labels = []
        self.max_length = max_length

        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith('.txt'):
                        self.files.append(os.path.join(label_dir, file))
                        self.labels.append(label)

        if not self.files:
            raise ValueError("No data files found in the specified directory.")

        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.labels)
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(self.labels)

        if button_encoder is None:
            self.button_encoder = LabelEncoder()
        else:
            self.button_encoder = button_encoder

        self.scaler = StandardScaler()
        self._initialize_scaler()

    def _initialize_scaler(self):
        sample_files = np.random.choice(self.files, min(100, len(self.files)), replace=False)
        data_samples = []
        for file in tqdm(sample_files, desc="Initializing scaler"):
            try:
                df = pd.read_csv(file, header=None, dtype={0: float, 1: float, 2: float, 3: float, 4: float, 5: str})
                data_samples.append(df[[0, 1, 2]])
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        if data_samples:
            data_samples = pd.concat(data_samples, axis=0)
            self.scaler.fit(data_samples)
        else:
            print("Warning: No valid data samples for scaler initialization.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]
        try:
            df = pd.read_csv(file, header=None, dtype={0: float, 1: float, 2: float, 3: float, 4: float, 5: str})
            df[5] = self.button_encoder.transform(df[5])  
            df[[0, 1, 2]] = self.scaler.transform(df[[0, 1, 2]])  
            if len(df) > self.max_length:
                df = df[:self.max_length]
            data = torch.tensor(df.values, dtype=torch.float32)
            if torch.isnan(data).any() or torch.isinf(data).any():
                raise ValueError(f"NaN or infinite values found in file {file}")
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            data = torch.zeros((self.max_length, 6), dtype=torch.float32)
        return data, label
      
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, labels
  
data_dir = '' # Put dir to main folder
dataset = OsuReplayDataset(data_dir)
print(f"Loaded {len(dataset)} samples from {data_dir}")

def stratified_split(dataset, test_size=0.2):
    df = pd.DataFrame({'files': dataset.files, 'labels': dataset.labels})
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['labels'])

    train_dataset = Subset(dataset, train_df.index)
    test_dataset = Subset(dataset, test_df.index)

    return train_dataset, test_dataset

train_dataset, test_dataset = stratified_split(dataset)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=16)#Carefull with num_workers as 16 will not work for most cpus
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=16)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 6
hidden_size = 88
output_size = len(os.listdir(data_dir))
num_layers = 4

model = BiLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)  
print(f"Model is on device: {next(model.parameters()).device}")
class_counts = [1370, 201787, 13277, 503, 1447] #IMPORTANT change these ratios depeniding on ammount of data per folder
total_samples = sum(class_counts)
class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  
accumulation_steps = 2
num_epochs = 200
max_grad_norm = 0.5  
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
        for i, (sequences, labels) in pbar:
            sequences, labels = sequences.to(device), labels.to(device)  
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            pbar.set_postfix({'Loss': running_loss / (i + 1)})
    epoch_loss = running_loss * accumulation_steps / len(train_dataloader)  
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for sequences, labels in test_dataloader:
            sequences, labels = sequences.to(device), labels.to(device) 
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_dataloader)
    test_losses.append(test_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}')
    with torch.no_grad():
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Gradient Norm: {total_norm:.4f}')
    if epoch > 1 and test_losses[-1] > test_losses[-2]:
        print("Warning: Test loss increased. Potential overfitting detected.")

torch.save(model.state_dict(), 'ProtoModel1.pth')
