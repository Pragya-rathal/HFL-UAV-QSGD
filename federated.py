from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch import nn

from clustering import Cluster
from compression import qsgd_compress, topk_error_feedback
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
        method: str = "F",
        quorum_fraction: float = 0.7,
        topk_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.global_model = global_model
        self.devices = devices
        self.device_loaders = device_loaders
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.global_rounds = global_rounds
        self.compression_bits = compression_bits
        self.method = method.upper()
        self.quorum_fraction = min(max(quorum_fraction, 0.0), 1.0)
        self.topk_ratio = min(max(topk_ratio, 1e-6), 1.0)
        self.rng = random.Random(seed)

        self.criterion = nn.CrossEntropyLoss()
        self.history: Dict[str, List[float]] = defaultdict(list)

        self.clusters: List[Cluster] = []
        self.device_to_cluster: Dict[int, int] = {}
        for cid in range(max(1, num_clusters)):
            members = [d for d in self.devices if d.device_id % max(1, num_clusters) == cid]
            self.clusters.append(Cluster(cluster_id=cid, devices=members))
            for d in members:
                self.device_to_cluster[d.device_id] = cid

        self.loader_by_id = {d.device_id: l for d, l in zip(self.devices, self.device_loaders)}
        self.data_size_by_id = {
            d.device_id: int(len(self.loader_by_id[d.device_id].dataset)) for d in self.devices
        }

        # For Top-K+Error Feedback.
        self.residuals: Dict[int, torch.Tensor | None] = {d.device_id: None for d in self.devices}

        # For quorum rotation (avoid always selecting the same participants).
        self.selection_count: Dict[int, int] = {d.device_id: 0 for d in self.devices}

    # Method A: No compression, no quorum.
    def _method_a(self, update: torch.Tensor, device_id: int) -> Tuple[torch.Tensor, float]:
        del device_id
        size_mb = (update.numel() * 32) / (8.0 * 1024.0 * 1024.0)
        return update, size_mb

    # Method B: Identity path (reserved variant baseline).
    def _method_b(self, update: torch.Tensor, device_id: int) -> Tuple[torch.Tensor, float]:
        return self._method_a(update, device_id)

    # Method C: Top-K + Error Feedback.
    def _method_c(self, update: torch.Tensor, device_id: int) -> Tuple[torch.Tensor, float]:
        compressed, residual, size_mb = topk_error_feedback(
            update=update,
            residual=self.residuals.get(device_id),
            ratio=self.topk_ratio,
        )
        self.residuals[device_id] = residual
        return compressed, size_mb

    # Method D: QSGD.
    def _method_d(self, update: torch.Tensor, device_id: int) -> Tuple[torch.Tensor, float]:
        del device_id
        levels = max(2, int(2**max(1, self.compression_bits)))
        compressed, size_mb = qsgd_compress(update, s=levels)
        return compressed, size_mb

    # Methods E/F use quorum; E uses C-compression and F uses D-compression.
    def _apply_compression(self, update: torch.Tensor, device_id: int) -> Tuple[torch.Tensor, float]:
        if self.method == "A":
            return self._method_a(update, device_id)
        if self.method == "B":
            return self._method_b(update, device_id)
        if self.method in {"C", "E"}:
            return self._method_c(update, device_id)
        # D/F default path.
        return self._method_d(update, device_id)

    def _select_quorum(self, device_pool: List[Device]) -> List[Device]:
        if self.method not in {"E", "F"}:
            return list(device_pool)

        if not device_pool:
            return []

        n_select = max(1, int(self.quorum_fraction * len(device_pool)))

        scored: List[Tuple[float, Device]] = []
        for d in device_pool:
            compute_power = float(getattr(d, "compute_power", 1.0))
            bandwidth = float(getattr(d, "bandwidth", getattr(d, "bandwidth_mbps", 1.0)))
            noise = self.rng.uniform(-0.05, 0.05)
            base_score = 0.7 * compute_power + 0.3 * bandwidth + noise

            # Rotation bonus for less frequently selected devices.
            rotation_bonus = 0.25 / (1.0 + self.selection_count[d.device_id])
            score = base_score + rotation_bonus
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [d for _, d in scored[:n_select]]

        for d in selected:
            self.selection_count[d.device_id] += 1

        return selected

    def train_round(self):
        # 1) Broadcast global model is implicit: each device receives self.global_model in train_local.
        # 2) Devices train.
        # 3) Compute updates.
        # 4) Apply compression + quorum (depending on method).
        # 5) Cluster aggregation.
        # 6) UAV aggregation: FedAvg weighted by data size.

        selected_devices = self._select_quorum(self.devices)
        selected_ids = {d.device_id for d in selected_devices}

        cluster_payloads: Dict[int, List[Tuple[torch.Tensor, int]]] = defaultdict(list)
        cluster_latencies: Dict[int, List[float]] = defaultdict(list)
        total_communication_mb = 0.0

        for dev in self.devices:
            if dev.device_id not in selected_ids:
                continue

            loader = self.loader_by_id[dev.device_id]
            dev.train_local(self.global_model, self.local_epochs, loader)

            update_out = dev.compute_update(self.global_model)
            if isinstance(update_out, tuple):
                raw_update = update_out[0]
                device_latency = float(update_out[1]) if len(update_out) > 1 else 0.0
            else:
                raw_update = update_out
                device_latency = 0.0

            compressed_update, communication_mb = self._apply_compression(
                update=raw_update,
                device_id=dev.device_id,
            )
            total_communication_mb += float(communication_mb)

            latency = dev.compute_latency(
                update_size_MB=float(communication_mb),
                num_batches=int(getattr(dev, "last_num_batches", len(loader))),
                epochs=self.local_epochs,
            )
            if device_latency > 0.0:
                latency = max(latency, device_latency)

            cid = self.device_to_cluster.get(dev.device_id, 0)
            cluster_payloads[cid].append((compressed_update, self.data_size_by_id[dev.device_id]))
            cluster_latencies[cid].append(latency)

        # cluster latency = max(device latency), round latency = max(cluster latencies)
        cluster_max_latency = {
            cid: (max(vals) if vals else 0.0) for cid, vals in cluster_latencies.items()
        }
        round_latency = max(cluster_max_latency.values()) if cluster_max_latency else 0.0

        # 5) Cluster aggregation.
        cluster_updates: List[torch.Tensor] = []
        cluster_weights: List[int] = []
        for cluster in self.clusters:
            payloads = cluster_payloads.get(cluster.cluster_id, [])
            if not payloads:
                continue
            updates = [u for u, _ in payloads]
            cluster_update = cluster.aggregate(updates)
            cluster_data_size = sum(w for _, w in payloads)
            cluster_updates.append(cluster_update)
            cluster_weights.append(cluster_data_size)

        if not cluster_updates:
            self.history["latency"].append(float(round_latency))
            self.history["communication"].append(float(total_communication_mb))
            self.history["active_devices"].append(float(len(selected_devices)))
            return float(round_latency), float(total_communication_mb), len(selected_devices)

        # 6) UAV aggregation: FedAvg weighted by data size.
        weight_sum = float(sum(cluster_weights))
        normalized = [w / weight_sum for w in cluster_weights]
        global_update = sum(w * u for w, u in zip(normalized, cluster_updates))

        current_flat = flatten_model_params(self.global_model)
        new_flat = current_flat + global_update
        assign_flat_params(self.global_model, new_flat)

        self.history["latency"].append(float(round_latency))
        self.history["communication"].append(float(total_communication_mb))
        self.history["active_devices"].append(float(len(selected_devices)))
        return float(round_latency), float(total_communication_mb), len(selected_devices)

    def run(self):
        for _ in range(self.global_rounds):
            self.train_round()
            metrics = evaluate(self.global_model, self.test_loader, self.criterion)
            self.history["loss"].append(metrics["loss"])
            self.history["accuracy"].append(metrics["accuracy"])

        return dict(self.history)
