"""Neural network models and parameter utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MNISTModel(nn.Module):
    """CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CIFAR10Model(nn.Module):
    """CNN for CIFAR-10 classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_model(dataset: str) -> nn.Module:
    if dataset == "mnist":
        return MNISTModel()
    elif dataset == "cifar10":
        return CIFAR10Model()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def flatten_model(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single 1D tensor."""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


def load_model(model: nn.Module, flat_tensor: torch.Tensor) -> None:
    """Load flattened parameters into model."""
    idx = 0
    for param in model.parameters():
        num_params = param.numel()
        param.data.copy_(flat_tensor[idx:idx + num_params].view(param.shape))
        idx += num_params


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module, bits_per_param: int = 32) -> float:
    """Get model size in megabytes."""
    num_params = count_parameters(model)
    return (num_params * bits_per_param) / (8 * 1024 * 1024)
