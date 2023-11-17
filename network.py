import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    def __init__(self, num_channels= 3, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        
    def forward(self, x):
        x = f.relu(self.conv1(x)) # X = 28x28 -> 26
        x = self.pool(x)         # x = 26x26 -> 13
        x = f.relu(self.conv2(x)) # x = 13x13 -> 11
        x = self.pool(x)         # x = 11x11 -> 5
        x = f.relu(self.conv3(x)) # x = 5x5   -> 3
        print(x.shape)
        x = x.view(-1, 3 * 3 * 64)  # Maintain spatial dimensions and convert to single dimension
        x = x.view(-1, 8192)  # Resize to be compatible with fully connected layer
        x = self.fc(x)
        return x
