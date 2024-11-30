import torch.nn as nn
from torchvision.models import resnet18, resnet34


class DeepFakeMethodClassifier(nn.Module):
    def __init__(self, num_classes=4, input_channels=1, cnn_out_channels=64):
        super(DeepFakeMethodClassifier, self).__init__()

        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, cnn_out_channels, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        self.fc = nn.Linear(self.resnet[-1][-1].bn2.num_features, num_classes)

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.resnet(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x
