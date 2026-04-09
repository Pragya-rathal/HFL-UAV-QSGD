import copy
from dataclasses import dataclass

import torch
from torch import nn

from model import model_update


@dataclass
class Device:
    device_id: int
    lr: float
    bandwidth_mbps: float = 10.0
    compute_factor: float = 1.0

    def __post_init__(self) -> None:
        self.local_model: nn.Module | None = None
        self.last_num_batches: int = 0

    def train_local(self, model: nn.Module, epochs: int, dataloader) -> nn.Module:
        self.local_model = copy.deepcopy(model)
        self.local_model.train()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.last_num_batches = len(dataloader)
        for _ in range(epochs):
            for xb, yb in dataloader:
                optimizer.zero_grad()
                logits = self.local_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return self.local_model

    def compute_update(self, global_model: nn.Module) -> torch.Tensor:
        if self.local_model is None:
            raise RuntimeError("Local model unavailable. Call train_local first.")
        return model_update(self.local_model, global_model)

    def compute_latency(self, update_size_MB: float, num_batches: int, epochs: int) -> float:
        transfer_seconds = (update_size_MB * 8.0) / max(self.bandwidth_mbps, 1e-6)
        compute_seconds = (num_batches * epochs) / max(self.compute_factor, 1e-6) * 0.01
        return transfer_seconds + compute_seconds
