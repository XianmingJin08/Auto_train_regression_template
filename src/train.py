import torch
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch import nn
import copy

def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (train_data, target) in enumerate(train_loader):
        train_data, target = train_data.float().to(device), target.to(device)
        optimizer.zero_grad()
        output = model(train_data)
        
        criterion = nn.MSELoss()  # Replace with MSE loss
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    return avg_loss


def validate(model, device, valid_loader):
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for train_data, target in valid_loader:
            train_data, target = train_data.float().to(device), target.to(device)
            output = model(train_data)
            loss = F.mse_loss(output, target, reduction='sum')
            val_loss += loss.item()
            val_preds.extend(output.tolist())
            val_targets.extend(target.tolist())

    val_loss /= len(valid_loader.dataset)
    val_mse = mean_squared_error(val_targets, val_preds)
    val_mae = mean_absolute_error(val_targets, val_preds)
    return val_loss, val_mse, val_mae


def test(model, device, test_loader):
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for train_data, target in test_loader:
            train_data, target = train_data.float().to(device), target.to(device)
            output = model(train_data)
            test_preds.extend(output.tolist())
            test_targets.extend(target.tolist())

    # Calculate metrics
    test_mse = mean_squared_error(test_targets, test_preds)
    return test_mse


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model