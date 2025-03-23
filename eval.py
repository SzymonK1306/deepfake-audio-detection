import os

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchaudio.transforms import MelSpectrogram

from torch.utils.tensorboard import SummaryWriter

from model import DeepFakeClassifier_no_transform

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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


def check_status(identifier):
    with open('trial_metadata.txt', 'r') as file:
        for line in file:
            if identifier in line:
                if "bonafide" in line:
                    return "bonafide"
                elif "spoof" in line:
                    return "spoof"
                else:
                    return "Unknown status"
    return "Identifier not found"


class AudioFileDataset(Dataset):
    def __init__(self, root_dir, target_length=16000, n_mels=64):
        self.file_paths = []
        self.labels = []
        self.target_length = target_length
        self.mel_transform = MelSpectrogram(n_mels=n_mels, f_min=100)

        # -------MISIOWY-------------------------------------------------------
        # with open('petrichorwq-DECRO-dataset-6fc9884\\en_train.txt', 'r') as file:
        #     for line in file:
        #         filename = 'petrichorwq_norm\\en_train\\' + line.split()[1] + '.flac'
        #         self.file_paths.append(filename)
        #         if 'bonafide' in line:
        #             self.labels.append(0)
        #         else:
        #             self.labels.append(1)
        # with open('petrichorwq-DECRO-dataset-6fc9884\\en_eval.txt', 'r') as file:
        #     for line in file:
        #         filename = 'petrichorwq_norm\\en_eval\\' + line.split()[1] + '.flac'
        #         self.file_paths.append(filename)
        #         if 'bonafide' in line:
        #             self.labels.append(0)
        #         else:
        #             self.labels.append(1)
        # with open('petrichorwq-DECRO-dataset-6fc9884\\en_dev.txt', 'r') as file:
        #     for line in file:
        #         filename = 'petrichorwq_norm\\en_dev\\' + line.split()[1] + '.flac'
        #         self.file_paths.append(filename)
        #         if 'bonafide' in line:
        #             self.labels.append(0)
        #         else:
        #             self.labels.append(1)

        # with open('trial_metadata.txt', 'r') as file:
        #     for line in file:
        #         filename = 'ASV_norm\\flac\\' + line.split()[1] + '.flac'
        #         self.file_paths.append(filename)
        #         if 'bonafide' in line:
        #             self.labels.append(0)
        #         else:
        #             self.labels.append(1)

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
        waveform, self.target_sr = torchaudio.load(self.file_paths[idx])
        waveform = self.add_white_noise(waveform, 20)
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or truncate waveform
        waveform = self.pad_or_truncate(waveform)

        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)

        # mel_spec = torch.nn.functional.normalize(mel_spec)
        mel_spec = torch.log(mel_spec + 1e-9)

        # Return mel-spectrogram and label
        return mel_spec, self.labels[idx]

    def add_white_noise(self, waveform, snr=20):
        """
        Adds white noise to the waveform while maintaining a target SNR.
        Args:
            waveform (torch.Tensor): Input waveform of shape (1, samples).
            snr (float): Signal-to-Noise Ratio in decibels (default 20 dB).
        Returns:
            torch.Tensor: Waveform with added white noise.
        """
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        white_noise = torch.rand_like(waveform) * 2 - 1  # Uniform distribution in [-1, 1]
        white_noise = white_noise * torch.sqrt(noise_power / torch.mean(white_noise ** 2))
        return waveform + white_noise

    def add_pink_noise(self, waveform, snr=20):
        """
        Adds pink noise to the waveform while maintaining a target SNR.
        Args:
            waveform (torch.Tensor): Input waveform of shape (1, samples).
            snr (float): Signal-to-Noise Ratio in decibels (default 20 dB).
        Returns:
            torch.Tensor: Waveform with added pink noise.
        """
        length = waveform.size(-1)
        # Generate white noise
        white_noise = torch.randn(length, device=waveform.device)
        # Apply FFT and scale amplitudes to follow 1/f
        fft_coeffs = torch.fft.rfft(white_noise)
        freqs = torch.fft.rfftfreq(length, d=1.0 / self.target_sr)
        fft_coeffs /= torch.sqrt(freqs + 1e-8)  # Scale amplitudes
        pink_noise = torch.fft.irfft(fft_coeffs, n=length)
        pink_noise = pink_noise.unsqueeze(0)  # Match waveform shape

        # Scale pink noise to match target SNR
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        pink_noise = pink_noise * torch.sqrt(noise_power / torch.mean(pink_noise ** 2))
        return waveform + pink_noise


root_dir = "dataset"

max_length = 338548

# for label, class_name in enumerate(os.listdir(root_dir)):
#     class_dir = os.path.join(root_dir, class_name)
#     if os.path.isdir(class_dir):
#         for root, _, files in os.walk(class_dir):
#             for file_name in files:
#                 if file_name.endswith('.flac'):
#                     waveform, sr = torchaudio.load(os.path.join(root, file_name))
#
#                     # Calculate the waveform length in samples
#                     length = waveform.size(1)
#                     if length > max_length:
#                         max_length = length

print(max_length)
max_length = max_length
dataset = AudioFileDataset(root_dir=root_dir, target_length=max_length, n_mels=64)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def check_class_distribution(loader):
    counts = [0, 0]
    for _, labels in loader:
        counts[0] += (labels == 0).sum().item()
        counts[1] += (labels == 1).sum().item()
    return counts


print("Train class distribution:", check_class_distribution(dataloader))

if torch.cuda.is_available():
    device = 'cuda'
else:
    print("CUDA unsupported")

model = DeepFakeClassifier_no_transform().to(device)
model.load_state_dict(torch.load('resnet_18.pth'))

criterion = nn.CrossEntropyLoss()

# Test evaluation
model.eval()
test_loss = 0.0
test_correct_predictions = 0
test_total_predictions = 0
test_outputs = []
test_labels = []
test_outputs_matrix = []
test_labels_matrix = []

with torch.no_grad():
    for mel_specs, labels in dataloader:
        mel_specs, labels = mel_specs.to(device), labels.to(device)
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * mel_specs.size(0)
        _, preds = torch.max(outputs, 1)
        test_correct_predictions += (preds == labels).sum().item()
        test_total_predictions += labels.size(0)

        test_outputs.append(outputs)
        test_labels.append(labels)

        test_outputs_matrix.extend(preds.cpu().numpy())
        test_labels_matrix.extend(labels.cpu().numpy())

cm = confusion_matrix(test_labels_matrix, test_outputs_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(cm.shape[0]), yticklabels=range(cm.shape[0]))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Test metrics
test_loss /= len(dataset)
test_accuracy = test_correct_predictions / test_total_predictions
test_precision, test_recall, test_f1 = calculate_epoch_metrics(test_outputs, test_labels)

print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, '
      f'Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}')


