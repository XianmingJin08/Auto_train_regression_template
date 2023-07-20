import torch
from torch import nn
import torch.nn.functional as F

class CNNLSTMModel(nn.Module):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()
        # in_channels = K, number of features
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)  # 1D Convolution
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(64, 32, batch_first=True)  # LSTM

        self.fc = nn.Sequential(
            nn.Linear(32, 16),  # adjust here
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )


    def forward(self, x):
        x = x.permute(0, 2, 1)  # Make sure the feature dimension is at the right place for Conv1d
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Change the dimensions back for LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Select the last output of the LSTM

        x = self.fc(x)

        return x
