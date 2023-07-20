import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
import pickle


def initialize_model(model, device, learning_rate):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def normalize_data(X):
    N, T, K = X.shape
    X = X.reshape(-1, T * K)
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    X = (X - mean) / (std + 1e-7)
    X = X.reshape(-1, T, K)
    return X


def normalize_data_scaler(X, load_path=None, save_path=None):
    N, T, K = X.shape
    X_reshaped = X.reshape(-1, T * K)

    if load_path:
        with open(load_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler()
        scaler.fit(X_reshaped)
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(scaler, f)
    X_norm = scaler.transform(X_reshaped).reshape(-1, T, K)

    return X_norm


class EarlyStopping:
    """Early stops the training if validation MSE doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_mse = float('inf')
        self.early_stop = False

    def __call__(self, val_mse, model, model_path):
        if val_mse < self.best_mse:
            self.best_mse = val_mse
            self.save_checkpoint(val_mse, model, model_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_mse, model, model_path):
        if self.verbose:
            print(f'Validation MSE decreased ({self.best_mse:.6f} --> {val_mse:.6f}). Saving model ...')
        torch.save(model.state_dict(), model_path)