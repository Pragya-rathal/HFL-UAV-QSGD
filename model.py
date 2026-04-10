import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, dataset: str = "mnist", num_classes: int = 10):
        super().__init__()
        ds = dataset.lower()
        if ds not in {"mnist", "cifar10", "cifar-10"}:
            raise ValueError("dataset must be 'mnist' or 'cifar10'")

        if ds == "mnist":
            in_channels = 1
            flattened_dim = 64 * 7 * 7
        else:
            in_channels = 3
            flattened_dim = 64 * 8 * 8

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_model_size_MB(self) -> float:
        total_bits = self.get_num_params() * 32
        return total_bits / (8 * 1024 * 1024)


# Backward-compatible alias for existing imports in the scaffold.
SimpleMLP = CNNClassifier


def flatten_model_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def load_model_params_from_flat(model: nn.Module, flat_tensor: torch.Tensor) -> None:
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        chunk = flat_tensor[offset : offset + numel].view_as(param)
        param.data.copy_(chunk)
        offset += numel


# Backward-compatible alias for existing imports in the scaffold.
assign_flat_params = load_model_params_from_flat


def model_update(local_model: nn.Module, global_model: nn.Module) -> torch.Tensor:
    local_flat = flatten_model_params(local_model)
    global_flat = flatten_model_params(global_model)
    return local_flat - global_flat
