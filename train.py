import os

import torch
import torchaudio
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchaudio.transforms import MelSpectrogram

from torch.utils.tensorboard import SummaryWriter

from model import DeepFakeClassifier

from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_epoch_metrics(all_outputs, all_labels):
    # Convert tensors to numpy arrays for sklearn compatibility
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # Get the predicted class by taking the max logit/probability
    preds = all_outputs.argmax(axis=1)

    # Calculate metrics
    precision = precision_score(all_labels, preds, average='weighted')
    recall = recall_score(all_labels, preds, average='weighted')
    f1 = f1_score(all_labels, preds, average='weighted')

    return precision, recall, f1


class AudioFileDataset(Dataset):
    def __init__(self, root_dir, target_length=16000, n_mels=64):
        self.file_paths = []
        self.labels = []
        self.target_length = target_length
        self.mel_transform = MelSpectrogram(n_mels=n_mels, f_min=100)

        # Load file paths and labels
        for label, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for root, _, files in os.walk(class_dir):
                    for file_name in files:
                        if file_name.endswith('.flac'):
                            self.file_paths.append(os.path.join(root, file_name))
                            self.labels.append(label)

    def pad_or_truncate(self, waveform):
        # Pad or truncate waveform to the target length
        if waveform.size(1) < self.target_length:
            padding = self.target_length - waveform.size(1)
            waveform = F.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.target_length]
        return waveform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load waveform and label
        waveform, sr = torchaudio.load(self.file_paths[idx])

        # Pad or truncate waveform
        waveform = self.pad_or_truncate(waveform)

        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)

        mel_spec = torch.nn.functional.normalize(mel_spec)

        # Return mel-spectrogram and label
        return mel_spec, self.labels[idx]


root_dir = "dataset/"

max_length = 0

for label, class_name in enumerate(os.listdir(root_dir)):
    class_dir = os.path.join(root_dir, class_name)
    if os.path.isdir(class_dir):
        for root, _, files in os.walk(class_dir):
            for file_name in files:
                if file_name.endswith('.flac'):
                    waveform, sr = torchaudio.load(os.path.join(root, file_name))

                    # Calculate the waveform length in samples
                    length = waveform.size(1)
                    if length > max_length:
                        max_length = length

print(max_length)

dataset = AudioFileDataset(root_dir=root_dir, target_length=max_length, n_mels=64)

# Split dataset into train, val, and test sets with stratification
labels = dataset.labels
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(dataset.file_paths, labels))
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(sss_val.split([dataset.file_paths[i] for i in train_idx],
                                        [labels[i] for i in train_idx]))

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Hyperparameters
learning_rate = 0.001
num_epochs = 20

model = DeepFakeClassifier().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter()    # tensorboard writer

print(f'Starting training using {device}')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_outputs = []
    all_labels = []

    # Training
    for batch_idx, (mel_specs, labels) in enumerate(train_loader):
        mel_specs, labels = mel_specs.to(device), labels.to(device)
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect metrics for epoch
        running_loss += loss.item() * mel_specs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)
        all_outputs.append(outputs)
        all_labels.append(labels)

    # Epoch metrics for training
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / total_predictions
    precision, recall, f1 = calculate_epoch_metrics(all_outputs, all_labels)
    writer.add_scalar("Train/Loss", epoch_loss, epoch)
    writer.add_scalar("Train/Accuracy", epoch_accuracy, epoch)
    writer.add_scalar("Train/Precision", precision, epoch)
    writer.add_scalar("Train/Recall", recall, epoch)
    writer.add_scalar("Train/F1 Score", f1, epoch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
    val_outputs = []
    val_labels = []

    with torch.no_grad():
        for mel_specs, labels in val_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * mel_specs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct_predictions += (preds == labels).sum().item()
            val_total_predictions += labels.size(0)
            val_outputs.append(outputs)
            val_labels.append(labels)

    # Validation metrics
    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct_predictions / val_total_predictions
    val_precision, val_recall, val_f1 = calculate_epoch_metrics(val_outputs, val_labels)
    writer.add_scalar("Validation/Loss", val_loss, epoch)
    writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)
    writer.add_scalar("Validation/Precision", val_precision, epoch)
    writer.add_scalar("Validation/Recall", val_recall, epoch)
    writer.add_scalar("Validation/F1 Score", val_f1, epoch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    # Save checkpoint
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

writer.close()

