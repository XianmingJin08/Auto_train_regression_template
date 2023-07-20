import torch
from torch import nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(6, 32, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(32, 64),  # adjust here
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)

        return x

