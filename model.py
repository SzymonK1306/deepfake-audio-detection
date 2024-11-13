import torch.nn as nn


class DeepFakeClassifier(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, cnn_out_channels=64, transformer_dim=256, num_heads=4,
                 num_layers=4):
        super(DeepFakeClassifier, self).__init__()

        # CNN Layers to extract features
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(input_channels, cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(cnn_out_channels, cnn_out_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_out_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(cnn_out_channels * 2, cnn_out_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_out_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Transformer Encoder Layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads), num_layers=num_layers
        )

        # Linear Layer for final classification
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn_layers(x)

        x = x.flatten(2)

        x = x.permute(2, 0, 1)

        x = self.transformer(x)

        x = x[-1, :, :]

        x = self.fc(x)

        return x