import torch
import torch.nn as nn
import torch.nn.functional as F

class AMTENNet(nn.Module):
    def __init__(self):
        super(AMTENNet, self).__init__()

        # Initial conv layers
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        # Trace extraction block
        self.trace_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.trace_relu1 = nn.ReLU(inplace=False)
        self.trace_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.trace_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.trace_relu2 = nn.ReLU(inplace=False)
        self.trace_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.trace_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.trace_relu3 = nn.ReLU(inplace=False)
        self.trace_pool3 = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.trace_relu1(self.trace_conv1(x))
        x = self.trace_pool1(x)

        x = self.trace_relu2(self.trace_conv2(x))
        x = self.trace_pool2(x)

        x = self.trace_relu3(self.trace_conv3(x))
        x = self.trace_pool3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x



