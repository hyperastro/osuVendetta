import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.current_device()}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

class OsuReplayDataset(Dataset):
    def __init__(self, data_dir, label_encoder=None, button_encoder=None, segment_size=1000, is_test=False):
        self.data_dir = data_dir
        self.files = []
        self.labels = []
        self.segment_size = segment_size
        self.is_test = is_test

        if not is_test:
            for label in os.listdir(data_dir):
                label_dir = os.path.join(data_dir, label)
                if os.path.isdir(label_dir):
                    for file in os.listdir(label_dir):
                        if file.endswith('.txt'):
                            self.files.append(os.path.join(label_dir, file))
                            self.labels.append(label)
        else:
            self.files.append(data_dir)
            self.labels.append(os.path.basename(data_dir).split('_')[0])

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
            all_buttons = []
            for file in self.files:
                try:
                    df = pd.read_csv(file, header=None, usecols=[5], dtype=str, chunksize=10000)
                    for chunk in df:
                        all_buttons.extend(chunk[5].unique())
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
            self.button_encoder.fit(all_buttons)
        else:
            self.button_encoder = button_encoder

        self.scaler = StandardScaler()
        self._initialize_scaler()

    def _initialize_scaler(self):
        sample_files = np.random.choice(self.files, min(1000, len(self.files)), replace=False)
        data_samples = []
        for file in tqdm(sample_files, desc="Initializing scaler"):
            try:
                df = pd.read_csv(file, header=None, usecols=[0, 3, 4], dtype=float, chunksize=10000)
                for chunk in df:
                    data_samples.append(chunk)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        data_samples = pd.concat(data_samples, axis=0)
        self.scaler.fit(data_samples)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]

        try:
            df = pd.read_csv(file, header=None, dtype={0: float, 1: float, 2: float, 3: float, 4: float, 5: str})
            df[5] = self.button_encoder.transform(df[5])
            df[[0, 3, 4]] = self.scaler.transform(df[[0, 3, 4]])

            if len(df) > self.segment_size:
                start = np.random.randint(0, len(df) - self.segment_size)
                segment = df.iloc[start:start+self.segment_size].values
            else:
                segment = df.values

            if len(segment) < self.segment_size:
                padding = np.zeros((self.segment_size - len(segment), segment.shape[1]))
                segment = np.vstack((segment, padding))

            segment = torch.tensor(segment, dtype=torch.float32)

            if torch.isnan(segment).any() or torch.isinf(segment).any():
                raise ValueError(f"NaN or infinite values found in file {file}")

        except Exception as e:
            print(f"Error reading file {file}: {e}")
            segment = torch.zeros((self.segment_size, 6), dtype=torch.float32)
        return segment, label




def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, labels

def stratified_split(dataset, test_size=0.2):
    df = pd.DataFrame({'files': dataset.files, 'labels': dataset.labels})
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['labels'])
    train_dataset = Subset(dataset, train_df.index)
    test_dataset = Subset(dataset, test_df.index)
    return train_dataset, test_dataset

data_dir = '/mnt/4b55a907-bb3f-43f9-ad49-8c8f30f6a000/ParsedReplays'
dataset = OsuReplayDataset(data_dir)
print(f"Loaded {len(dataset)} samples from {data_dir}")

train_dataset, test_dataset = stratified_split(dataset)
train_dataloader = DataLoader(train_dataset, batch_size=180, shuffle=True, collate_fn=collate_fn, num_workers=16)
test_dataloader = DataLoader(test_dataset, batch_size=180, shuffle=False, collate_fn=collate_fn, num_workers=16)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

input_size = 6
hidden_size = 192
output_size = 2
num_layers = 2
model = BiLSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
print(f"Model is on device: {next(model.parameters()).device}")


class_counts = [13277, 13277]  #be carefull if you are training with diffrent num of files you have to ajust the num of files present in each class
total_samples = sum(class_counts)
class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
class_weights = torch.tensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005,weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


num_epochs = 30
max_grad_norm = 1.0
accumulation_steps = 2
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for i, (sequences, labels) in pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': running_loss / ((i + 1) * accumulation_steps)})

    train_losses.append(running_loss / len(train_dataloader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_dataloader):.4f}")
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sequences, labels in test_dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    test_losses.append(test_loss / len(test_dataloader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss / len(test_dataloader):.4f}")

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(output_size)),
                yticklabels=list(range(output_size)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_epoch_{epoch + 1}.png')
    plt.close()
    scheduler.step()
torch.save(model.state_dict(), 'bilstm_model.pth')
print("Model saved as 'bilstm_model.pth'")
dummy_input = torch.randn(1, 1000, input_size).to(device)
onnx_path = 'osu_anti_cheat_modelFINAL.onnx'
torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=['input'], output_names=['output'])
print(f"Model has been exported to ONNX format at {onnx_path}")
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Test Loss over Epochs')
plt.savefig('loss_curves.png')
plt.close()

