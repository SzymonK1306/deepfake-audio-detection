import os

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchaudio.transforms import MelSpectrogram

from torch.utils.tensorboard import SummaryWriter

from model import DeepFakeClassifier, DeepFakeClassifier_no_transform

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def calculate_epoch_metrics(all_outputs, all_labels):
    # Convert tensors to numpy arrays for sklearn compatibility
    all_outputs = torch.cat(all_outputs).cpu().detach().numpy()
    all_labels = torch.cat(all_labels).cpu().detach().numpy()

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
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for root, _, files in os.walk(class_dir):
                    for file_name in files:
                        if file_name.endswith('.flac'):
                            self.file_paths.append(os.path.join(root, file_name))
                            if 'REALSPEECH' in class_dir:
                                self.labels.append(0)
                            else:
                                self.labels.append(1)

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
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or truncate waveform
        waveform = self.pad_or_truncate(waveform)

        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)

        # mel_spec = torch.nn.functional.normalize(mel_spec)
        mel_spec = torch.log(mel_spec + 1e-9)

        # Return mel-spectrogram and label
        return mel_spec, self.labels[idx]


root_dir = "dataset"

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
max_length = max_length
dataset = AudioFileDataset(root_dir=root_dir, target_length=max_length, n_mels=64)

# Extract labels
labels = torch.tensor([label for _, label in dataset])

# Stratified split using sklearn
train_indices, temp_indices = train_test_split(
    range(len(labels)),
    test_size=0.3,
    stratify=labels,
    random_state=42
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.5,
    stratify=labels[temp_indices],
    random_state=42
)

# Create subsets
# train_dataset = Subset(dataset, train_indices)
# val_dataset = Subset(dataset, val_indices)
# test_dataset = Subset(dataset, test_indices)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Verify class distribution
def check_class_distribution(loader):
    counts = [0, 0]
    for _, labels in loader:
        counts[0] += (labels == 0).sum().item()
        counts[1] += (labels == 1).sum().item()
    return counts


print("Train class distribution:", check_class_distribution(train_loader))
print("Validation class distribution:", check_class_distribution(val_loader))
print("Test class distribution:", check_class_distribution(test_loader))

if torch.cuda.is_available():
    device = 'cuda'
else:
    print("CUDA unsupported")

# Hyperparameters
learning_rate = 0.0005
num_epochs = 20

model = DeepFakeClassifier_no_transform().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-5)


# Specify the model name
model_name = "resnet18"

# Set the log directory path
log_dir = f"runs/{model_name}"

writer = SummaryWriter(log_dir=log_dir)  # tensorboard writer

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
        optimizer.zero_grad()
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)

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
    # torch.save(model.state_dict(), f'checkpoint_epoch_{epoch + 1}.pth')

# Test evaluation
model.eval()
test_loss = 0.0
test_correct_predictions = 0
test_total_predictions = 0
test_outputs = []
test_labels = []

with torch.no_grad():
    for mel_specs, labels in test_loader:
        mel_specs, labels = mel_specs.to(device), labels.to(device)
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * mel_specs.size(0)
        _, preds = torch.max(outputs, 1)
        test_correct_predictions += (preds == labels).sum().item()
        test_total_predictions += labels.size(0)
        test_outputs.append(outputs)
        test_labels.append(labels)

# Test metrics
test_loss /= len(test_loader.dataset)
test_accuracy = test_correct_predictions / test_total_predictions
test_precision, test_recall, test_f1 = calculate_epoch_metrics(test_outputs, test_labels)
writer.add_scalar("Test/Loss", test_loss, num_epochs)
writer.add_scalar("Test/Accuracy", test_accuracy, num_epochs)
writer.add_scalar("Test/Precision", test_precision, num_epochs)
writer.add_scalar("Test/Recall", test_recall, num_epochs)
writer.add_scalar("Test/F1 Score", test_f1, num_epochs)

torch.save(model.state_dict(), f'resnet_18.pth')

print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, '
      f'Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}')
writer.close()

