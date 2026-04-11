"""
IoT Device Model for UAV-assisted Hierarchical Federated Learning.
Each device has physically meaningful parameters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IoTDevice:
    device_id: int
    compute_power: float          # FLOPS scaling [0.5, 2.0]
    bandwidth: float              # Mbps [1, 10]
    distance: float               # meters [10, 100]
    channel_quality: float        # 1/d – monotonic decreasing
    clustering_coefficient: float # [0.3, 1.0]
    score: float = 0.0
    cluster_id: int = -1
    is_cluster_head: bool = False
    residual_buffer: Optional[np.ndarray] = None  # for error feedback
    energy_used: float = 0.0      # optional tracking

    # ─── Latency helpers ────────────────────────────────────────────────────
    def compute_time(self, base_compute_time: float) -> float:
        """T_comp = base_compute_time / compute_power"""
        return base_compute_time / self.compute_power

    def comm_time(self, message_size_mb: float) -> float:
        """T_comm = message_size (MB) / bandwidth (Mbps) * 8 (bits/byte)"""
        return (message_size_mb * 8.0) / self.bandwidth   # seconds

    def total_time(self, base_compute_time: float, message_size_mb: float) -> float:
        return self.compute_time(base_compute_time) + self.comm_time(message_size_mb)


def create_devices(num_devices: int, seed: int) -> List[IoTDevice]:
    """
    Deterministically create a heterogeneous set of IoT devices.
    All physical parameters drawn from realistic distributions.
    """
    rng = np.random.RandomState(seed)

    devices = []
    for i in range(num_devices):
        distance = rng.uniform(10.0, 100.0)
        channel_quality = 1.0 / distance          # monotonic decreasing

        dev = IoTDevice(
            device_id=i,
            compute_power=rng.uniform(0.5, 2.0),
            bandwidth=rng.uniform(1.0, 10.0),
            distance=distance,
            channel_quality=channel_quality,
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
    """Assign cluster-head selection scores in-place (normalised)."""
    cp = np.array([d.compute_power for d in devices])
    cc = np.array([d.clustering_coefficient for d in devices])
    bw = np.array([d.bandwidth for d in devices])

    # normalise each dimension to [0,1]
    def norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    scores = w_compute * norm(cp) + w_clustering * norm(cc) + w_bandwidth * norm(bw)
    for dev, s in zip(devices, scores):
        dev.score = float(s)
