import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import LabelEncoder

class MyDataset(Dataset):
    def __init__(self, train_data, labels):
        self.train_data = train_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.train_data[idx], self.labels[idx]

def load_and_split_data(train_data, labels, valid_ratio=0.2, test_ratio=0.1):
    X = list(zip(train_data))
    y = labels

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_ratio, random_state=42)

    return (
        MyDataset(np.array([x[0] for x in X_train]), y_train),
        MyDataset(np.array([x[0] for x in X_valid]), y_valid),
        MyDataset(np.array([x[0] for x in X_test]), y_test),
    )

def get_dataloaders(train_data, labels, batch_size):
    train_dataset, valid_dataset, test_dataset = load_and_split_data(train_data, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

def get_all_dataloader (train_data, labels, batch_size):  # add data_C
    X = list(zip(train_data))
    y = labels

    dataset = MyDataset(np.array([x[0] for x in X]), y)  # add data_C
    all_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return all_dataloader