from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import random


@dataclass
class Device:
    device_id: int
    compute_power: float
    bandwidth_mbps: float
    distance_m: float
    channel_quality: float
    clustering_coefficient: float
    residual: float = 0.0
    last_latency_s: float = 0.0
    latency_history_s: List[float] = field(default_factory=list)

    @property
    def bandwidth(self) -> float:
        """Bandwidth alias retained for readability in formulas (Mbps)."""
        return self.bandwidth_mbps

    @property
    def distance(self) -> float:
        """Distance alias retained for readability in formulas (meters)."""
        return self.distance_m

    @property
    def avg_latency_s(self) -> float:
        if not self.latency_history_s:
            return 0.0
        return sum(self.latency_history_s) / len(self.latency_history_s)

    def compute_latency(self, base_compute_time: float, message_size_mb: float) -> float:
        """
        Compute end-to-end device latency with the required physical model:
          T_comp = base_compute_time / compute_power
          T_comm = message_size_mb / bandwidth
          T_device = T_comp + T_comm
        """
        safe_compute = max(self.compute_power, 1e-8)
        safe_bandwidth = max(self.bandwidth_mbps, 1e-8)

        t_comp = base_compute_time / safe_compute
        t_comm = message_size_mb / safe_bandwidth
        return t_comp + t_comm

    def record_latency(self, base_compute_time: float, message_size_mb: float) -> float:
        latency = self.compute_latency(base_compute_time=base_compute_time, message_size_mb=message_size_mb)
        self.last_latency_s = latency
        self.latency_history_s.append(latency)
        return latency


def compute_device_latency(device: Device, base_compute_time: float, message_size_mb: float) -> float:
    """Module-level helper to keep integration simple for existing pipelines."""
    return device.record_latency(base_compute_time=base_compute_time, message_size_mb=message_size_mb)


def generate_devices(num_devices: int, seed: int) -> Dict[int, Device]:
    rng = random.Random(seed)
    devices: Dict[int, Device] = {}
    for i in range(num_devices):
        distance = rng.uniform(10.0, 100.0)
        devices[i] = Device(
            device_id=i,
            compute_power=rng.uniform(0.5, 2.0),
            bandwidth_mbps=rng.uniform(1.0, 10.0),
            distance_m=distance,
            channel_quality=1.0 / distance,
            clustering_coefficient=rng.uniform(0.3, 1.0),
        )
    return devices
