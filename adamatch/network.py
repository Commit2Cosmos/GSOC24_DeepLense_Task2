import torch
from torch import nn
from torchvision.models import resnet34

FEATURES_SIZE = 256

class Encoder(nn.Module):
    def __init__(self, features_size=FEATURES_SIZE):
        """
        ResNet based neural network that receives images and encodes them into an array of size `features_size`.

        Arguments:
        ----------features_size: int
            Size of encoded features array.
        """

        super().__init__()
        
        self.resnet = resnet34()
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, features_size)

    def forward(self, x):
        x = self.resnet(x)
        return x


class Classifier(nn.Module):
    def __init__(self, features_size=FEATURES_SIZE, n_classes=1):
        """
        Neural network that receives an array of size `features_size` and classifies it into `n_classes` classes.

        Arguments:
        ----------
        features_size: int
            Size of encoded features array.

        n_classes: int
            Number of classes to classify the encoded array into.
        """

        super().__init__()
        self.fc = nn.Linear(features_size, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x