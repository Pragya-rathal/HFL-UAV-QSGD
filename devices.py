### devices.py
import numpy as np
import torch
import torch.nn as nn
import copy
from model import flatten_model, load_model, create_model


class IoTDevice:
    def __init__(self, device_id, config, data_loader, dataset_size, rng=None):
        self.device_id = device_id
        self.config = config
        self.data_loader = data_loader
        self.dataset_size = max(1, dataset_size)

        if rng is None:
            rng = np.random.RandomState(device_id)

        self.compute_power = rng.uniform(0.5, 2.0)
        self.bandwidth = rng.uniform(1.0, 10.0)  # Mbps
        self.distance = rng.uniform(10.0, 100.0)
        self.channel_quality = 1.0 / self.distance
        self.clustering_coefficient = rng.uniform(0.3, 1.0)

        self.residual = None  # for error feedback
        self.last_participated = -1  # round tracker for quorum rotation

        # Torch device
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(config).to(self.torch_device)

    def reset_residual(self, param_size):
        self.residual = torch.zeros(param_size)

    def train_local(self, global_flat, epochs=None):
        """Train locally starting from global_flat parameters."""
        if epochs is None:
            epochs = self.config['local_epochs']

        # Load global model
        load_model(self.model, global_flat.to(self.torch_device))
        self.model.train()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            for batch_x, batch_y in self.data_loader:
                batch_x = batch_x.to(self.torch_device)
                batch_y = batch_y.to(self.torch_device)
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        local_flat = flatten_model(self.model).cpu()
        update = local_flat - global_flat.cpu()
        num_batches = len(self.data_loader)
        return update, avg_loss, num_batches

    def compute_latency(self, num_batches, epochs, message_size_MB):
        base_compute_time = num_batches * epochs * 0.01
        T_comp = base_compute_time / self.compute_power
        T_comm = (message_size_MB * 8) / self.bandwidth
        T_device = T_comp + T_comm
        return T_device

    def quorum_score(self, noise_std=0.05, rng=None):
        if rng is None:
            noise = np.random.normal(0, noise_std)
        else:
            noise = rng.normal(0, noise_std)
        return 0.7 * self.compute_power + 0.3 * self.bandwidth + noise


def create_devices(config, device_loaders, dataset_sizes):
    rng = np.random.RandomState(config['seed'])
    devices = []
    for i in range(config['num_devices']):
        d = IoTDevice(i, config, device_loaders[i], dataset_sizes[i], rng=rng)
        devices.append(d)
    return devices
