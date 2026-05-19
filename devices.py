"""
IoT Device model for UAV-assisted Hierarchical Federated Learning.

These are physical-property dataclasses (compute power, bandwidth, distance, ...).
They aren't tensors — keeping them on the CPU is correct; moving them to GPU
would not help. Federated.py stores per-device residual *tensors* in its own
dict on the model's device.
"""

from dataclasses import dataclass
from typing import List, Optional, Any

import numpy as np


@dataclass
class IoTDevice:
    device_id: int
    compute_power: float          # FLOPS scaling [0.5, 2.0]
    bandwidth: float              # Mbps           [1, 10]
    distance: float               # meters         [10, 100]
    channel_quality: float        # 1/distance — monotonic decreasing
    clustering_coefficient: float # [0.3, 1.0]
    score: float = 0.0
    cluster_id: int = -1
    is_cluster_head: bool = False
    residual_buffer: Optional[Any] = None   # legacy field; unused by torch path
    energy_used: float = 0.0

    # ── Latency helpers ──────────────────────────────────────────────────────
    def compute_time(self, base_compute_time: float) -> float:
        """T_comp = base / compute_power"""
        return base_compute_time / self.compute_power

    def comm_time(self, message_size_mb: float) -> float:
        """T_comm = (MB · 8 bits/byte) / Mbps  →  seconds"""
        return (message_size_mb * 8.0) / self.bandwidth

    def total_time(self, base_compute_time: float, message_size_mb: float) -> float:
        return self.compute_time(base_compute_time) + self.comm_time(message_size_mb)


def create_devices(num_devices: int, seed: int) -> List[IoTDevice]:
    """Deterministic heterogeneous device population."""
    rng = np.random.RandomState(seed)
    devices: List[IoTDevice] = []
    for i in range(num_devices):
        distance = rng.uniform(10.0, 100.0)
        dev = IoTDevice(
            device_id=i,
            compute_power=rng.uniform(0.5, 2.0),
            bandwidth=rng.uniform(1.0, 10.0),
            distance=distance,
            channel_quality=1.0 / distance,
            clustering_coefficient=rng.uniform(0.3, 1.0),
        )
        devices.append(dev)
    return devices


def compute_device_scores(
    devices: List[IoTDevice],
    w_compute: float,
    w_clustering: float,
    w_bandwidth: float,
) -> None:
    """Min-max normalised weighted score for cluster-head selection. In-place."""
    cp = np.array([d.compute_power          for d in devices])
    cc = np.array([d.clustering_coefficient for d in devices])
    bw = np.array([d.bandwidth              for d in devices])

    def norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    scores = w_compute * norm(cp) + w_clustering * norm(cc) + w_bandwidth * norm(bw)
    for dev, s in zip(devices, scores):
        dev.score = float(s)
