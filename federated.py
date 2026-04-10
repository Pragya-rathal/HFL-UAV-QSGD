from collections import defaultdict
from typing import Dict, List

import torch
from torch import nn

from clustering import Cluster
from compression import compressed_size_mb, dequantize_tensor, quantize_tensor
from devices import Device
from metrics import evaluate
from model import assign_flat_params, flatten_model_params


class FederatedTrainer:
    def __init__(
        self,
        global_model: nn.Module,
        devices: List[Device],
        device_loaders,
        test_loader,
        num_clusters: int,
        local_epochs: int,
        global_rounds: int,
        compression_bits: int,
    ):
        self.global_model = global_model
        self.devices = devices
        self.device_loaders = device_loaders
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.global_rounds = global_rounds
        self.compression_bits = compression_bits
        self.criterion = nn.CrossEntropyLoss()
        self.history: Dict[str, List[float]] = defaultdict(list)

        self.clusters = [Cluster(cluster_id=i) for i in range(num_clusters)]
        self.device_to_cluster = {
            d.device_id: d.device_id % num_clusters for d in self.devices
        }

    def train_round(self):
        cluster_updates: Dict[int, List[torch.Tensor]] = defaultdict(list)
        latencies: List[float] = []

        for dev, loader in zip(self.devices, self.device_loaders):
            dev.train_local(self.global_model, self.local_epochs, loader)
            update = dev.compute_update(self.global_model)
            q, q_min, q_scale = quantize_tensor(update, self.compression_bits)
            restored = dequantize_tensor(q, q_min, q_scale)

            mb = compressed_size_mb(q, self.compression_bits)
            latency = dev.compute_latency(mb, dev.last_num_batches, self.local_epochs)
            latencies.append(latency)

            cid = self.device_to_cluster[dev.device_id]
            cluster_updates[cid].append(restored)

        aggregated_clusters = []
        for cluster in self.clusters:
            updates = cluster_updates.get(cluster.cluster_id, [])
            if updates:
                agg = cluster.aggregate(updates)
                aggregated_clusters.append(agg)

        if not aggregated_clusters:
            return 0.0

        global_update = torch.stack(aggregated_clusters, dim=0).mean(dim=0)
        current_flat = flatten_model_params(self.global_model)
        new_flat = current_flat + global_update
        assign_flat_params(self.global_model, new_flat)

        return float(sum(latencies) / len(latencies)) if latencies else 0.0

    def run(self):
        for _ in range(self.global_rounds):
            avg_latency = self.train_round()
            metrics = evaluate(self.global_model, self.test_loader, self.criterion)
            self.history["loss"].append(metrics["loss"])
            self.history["accuracy"].append(metrics["accuracy"])
            self.history["latency"].append(avg_latency)

        return dict(self.history)
