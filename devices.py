from dataclasses import dataclass
from typing import Dict
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
