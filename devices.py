"""Device representation and management."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from model import flatten_model, load_model, get_model_size_mb


@dataclass
class DeviceProperties:
    device_id: int
    compute_power: float
    bandwidth: float
    distance: float
    channel_quality: float
    clustering_coefficient: float = 0.5


@dataclass
class DeviceState:
    residual: Optional[torch.Tensor] = None
    last_participated_round: int = -1
    total_samples_trained: int = 0


class Device:
    """Represents an IoT device in the federated learning system."""
    
    def __init__(
        self,
        device_id: int,
        properties: DeviceProperties,
        data_loader: DataLoader,
        model_template: nn.Module,
        learning_rate: float = 0.01,
        torch_device: str = "cpu"
    ):
        self.device_id = device_id
        self.properties = properties
        self.data_loader = data_loader
        self.num_samples = len(data_loader.dataset)
        self.learning_rate = learning_rate
        self.torch_device = torch_device
        
        self.model = type(model_template)().to(torch_device)
        self.state = DeviceState()
    
    def train_local(
        self,
        global_params: torch.Tensor,
        num_epochs: int
    ) -> Tuple[torch.Tensor, int]:
        """Train locally and return update (local - global)."""
        load_model(self.model, global_params.clone())
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(num_epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.torch_device), target.to(self.torch_device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        local_params = flatten_model(self.model)
        update = local_params - global_params
        
        self.state.total_samples_trained += self.num_samples * num_epochs
        
        return update, self.num_samples
    
    def compute_latency(
        self,
        num_epochs: int,
        message_size_mb: float
    ) -> Tuple[float, float, float]:
        """Compute training and communication latency."""
        num_batches = len(self.data_loader)
        base_compute_time = num_batches * num_epochs * 0.01
        t_comp = base_compute_time / self.properties.compute_power
        
        t_comm = (message_size_mb * 8) / self.properties.bandwidth
        
        t_total = t_comp + t_comm
        
        return t_comp, t_comm, t_total


def create_devices(
    num_devices: int,
    data_loaders: Dict[int, DataLoader],
    model_template: nn.Module,
    config,
    seed: int
) -> List[Device]:
    """Create device instances with random properties."""
    np.random.seed(seed)
    
    devices = []
    device_cfg = config.device
    
    for i in range(num_devices):
        compute_power = np.random.uniform(*device_cfg.compute_power_range)
        bandwidth = np.random.uniform(*device_cfg.bandwidth_range)
        distance = np.random.uniform(*device_cfg.distance_range)
        channel_quality = 1.0 / distance
        clustering_coeff = np.random.uniform(*device_cfg.clustering_coeff_range)
        
        properties = DeviceProperties(
            device_id=i,
            compute_power=compute_power,
            bandwidth=bandwidth,
            distance=distance,
            channel_quality=channel_quality,
            clustering_coefficient=clustering_coeff
        )
        
        device = Device(
            device_id=i,
            properties=properties,
            data_loader=data_loaders[i],
            model_template=model_template,
            learning_rate=config.training.learning_rate
        )
        
        devices.append(device)
    
    return devices
