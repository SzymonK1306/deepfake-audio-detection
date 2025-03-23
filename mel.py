import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import pad

max_length = 1478

# filepath = 'dataset_22k\\BARK_22050Hz_normalized\\226\\226-131533-0014.flac'
# filepath = 'dataset_22k\\ElevenLabs_22050Hz_normalized\\226_Deb\\ElevenLabs226-Deb226-131533-0014.flac'
# filepath = 'dataset_22k\\OpenVoice_22050Hz_normalized\\226\\226-131533-0014_cloned.flac'
# filepath = 'dataset_22k\\Toucan_22050Hz_normalized\\226\\226-131533-0014.flac'
filepath = 'dataset\\RealSpeach\\4014\\4014-131533-0014.flac'
# Step 1: Load the .flac file
waveform, sample_rate = torchaudio.load(filepath)

# Step 2: Create a MelSpectrogram transform
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64, f_min=100)
mel_spectrogram = mel_spectrogram_transform(waveform)

# Normalize the mel-spectrogram
mel_spectrogram = torch.nn.functional.normalize(mel_spectrogram)
#
# # Pad to max_length if necessary
# if mel_spectrogram.shape[-1] < max_length:
#     mel_spectrogram = pad(mel_spectrogram, (0, max_length - mel_spectrogram.shape[-1]))
#
# # Trim if it's longer than max_length
# mel_spectrogram = mel_spectrogram[:, :, :max_length]

# Step 4: Plot the Mel Spectrogram (optional, for visualization)
# plt.figure(figsize=(10, 4))
# plt.imshow(mel_spectrogram.log2()[0], cmap='inferno', origin='lower', aspect='auto', extent=[0, mel_spectrogram.size(1), 0, mel_spectrogram.size(0)])
# plt.title('Mel Spectrogram')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.colorbar(format="%+2.0f dB")
# plt.show()


mel_spectrogram_db = T.AmplitudeToDB()(mel_spectrogram)
#
# Plot the Mel Spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram_db[0].numpy(), aspect='auto', origin='lower', cmap='inferno')
plt.colorbar(format="%+2.0f dB")
# plt.title("Mel-Spectrogram (dB)")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()