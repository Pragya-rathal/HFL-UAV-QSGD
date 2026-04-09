import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def flatten_model_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def assign_flat_params(model: nn.Module, flat_tensor: torch.Tensor) -> None:
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        chunk = flat_tensor[offset : offset + numel].view_as(param)
        param.data.copy_(chunk)
        offset += numel


def model_update(local_model: nn.Module, global_model: nn.Module) -> torch.Tensor:
    local_flat = flatten_model_params(local_model)
    global_flat = flatten_model_params(global_model)
    return local_flat - global_flat
