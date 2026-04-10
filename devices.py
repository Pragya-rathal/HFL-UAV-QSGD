import copy
import random
from dataclasses import dataclass

import torch
from torch import nn

from model import flatten_model_params


@dataclass
class Device:
    device_id: int
    lr: float = 0.01
    compute_power: float | None = None
    bandwidth: float | None = None  # Mbps
    distance: float | None = None

    def __post_init__(self) -> None:
        self.compute_power = (
            float(self.compute_power)
            if self.compute_power is not None
            else random.uniform(0.5, 2.0)
        )
        self.bandwidth = (
            float(self.bandwidth) if self.bandwidth is not None else random.uniform(1.0, 10.0)
        )
        self.distance = (
            float(self.distance) if self.distance is not None else random.uniform(10.0, 100.0)
        )

        self.compute_power = min(max(self.compute_power, 0.5), 2.0)
        self.bandwidth = min(max(self.bandwidth, 1.0), 10.0)
        self.distance = min(max(self.distance, 10.0), 100.0)

        self.channel_quality: float = 1.0 / self.distance
        self.local_model: nn.Module | None = None
        self.last_num_batches: int = 0
        self.last_epochs: int = 0
        self.residual_buffer: torch.Tensor | None = None

    def train_local(self, model: nn.Module, epochs: int, dataloader) -> nn.Module:
        self.local_model = copy.deepcopy(model)
        self.local_model.train()

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.last_num_batches = len(dataloader)
        self.last_epochs = epochs

        if self.residual_buffer is None:
            flat = flatten_model_params(self.local_model)
            self.residual_buffer = torch.zeros_like(flat)

        for _ in range(epochs):
            for xb, yb in dataloader:
                optimizer.zero_grad()
                logits = self.local_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return self.local_model

    def compute_update(self, global_model: nn.Module):
        if self.local_model is None:
            raise RuntimeError("Local model unavailable. Call train_local first.")

        local_flat = flatten_model_params(self.local_model)
        global_flat = flatten_model_params(global_model)

        if self.residual_buffer is None or self.residual_buffer.numel() != local_flat.numel():
            self.residual_buffer = torch.zeros_like(local_flat)

        update = (local_flat - global_flat) + self.residual_buffer

        communication_size_mb = (update.numel() * 32) / (8.0 * 1024.0 * 1024.0)
        latency = self.compute_latency(
            update_size_MB=communication_size_mb,
            num_batches=self.last_num_batches,
            epochs=self.last_epochs,
        )

        return update, latency, communication_size_mb

    def compute_latency(self, update_size_MB: float, num_batches: int, epochs: int) -> float:
        base_compute_time = num_batches * epochs * 0.01
        t_comp = base_compute_time / self.compute_power
        t_comm = (update_size_MB * 8.0) / self.bandwidth
        return t_comp + t_comm
