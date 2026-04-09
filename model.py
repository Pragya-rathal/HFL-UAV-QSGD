from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


@dataclass(eq=False)
class DummyCNN(nn.Module):
    in_channels: int
    num_classes: int
    input_size: int

    def __post_init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        spatial = self.input_size // 4
        self.fc1 = nn.Linear(64 * spatial * spatial, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    @classmethod
    def create(cls, dataset: str = "MNIST") -> "DummyCNN":
        key = dataset.replace("-", "").replace("_", "").upper()
        if key == "MNIST":
            return cls(in_channels=1, num_classes=10, input_size=28)
        if key == "CIFAR10":
            return cls(in_channels=3, num_classes=10, input_size=32)
        raise ValueError(f"Unsupported dataset '{dataset}'. Use MNIST or CIFAR-10.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def state_vector(self) -> List[float]:
        with torch.no_grad():
            flat = [parameter.detach().view(-1).cpu() for parameter in self.parameters()]
            return torch.cat(flat).tolist() if flat else []

    def load_state_vector(self, vec: List[float]) -> None:
        with torch.no_grad():
            tensor_vec = torch.tensor(vec, dtype=torch.float32)
            expected = self.num_params()
            if tensor_vec.numel() != expected:
                raise ValueError(f"State length mismatch: got {tensor_vec.numel()}, expected {expected}.")

            offset = 0
            for parameter in self.parameters():
                size = parameter.numel()
                chunk = tensor_vec[offset : offset + size].view_as(parameter)
                parameter.copy_(chunk.to(parameter.device, dtype=parameter.dtype))
                offset += size
