import torch
from torch import nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()

        self.fc1 = nn.Linear(6, 64)  # Adjust the hidden size as needed
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 32)  # Adjust the hidden size as needed
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(32, 1)  # Output size is 1 for regression

    def forward(self, x):
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x