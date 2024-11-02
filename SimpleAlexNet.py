import torch
import torch.nn as nn

# Define a simplified AlexNet for MNIST
class SimpleAlexNet(nn.Module):
    def __init__(self):
        super(SimpleAlexNet, self).__init__()
        self.features = nn.Sequential(
            # First convolution layer, adjusted for grayscale input (1 channel)
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolution layer
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolution layer
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Pass a dummy input through the features part to get the correct size
        self.flatten_size = self._get_flatten_size()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.flatten_size, 1024),  # Use the calculated flatten size here
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def _get_flatten_size(self):
        # Forward a dummy input through the feature extractor to determine the output size
        with torch.no_grad():
            x = torch.randn(1, 1, 32, 32)  # Dummy batch size of 1 with 32x32 input
            x = self.features(x)
            return x.view(1, -1).size(1)  # Flatten and return the size

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.classifier(x)
        return x