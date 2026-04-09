from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Device:
    device_id: int
    compute_power: float
    bandwidth_mbps: float
    distance_m: float
    channel_quality: float
    clustering_coefficient: float
    x: float
    y: float
    energy_joules: float = 0.0



def generate_devices(num_devices: int, seed: int) -> Dict[int, Device]:
    rng = np.random.default_rng(seed)
    devices = {}
    for i in range(num_devices):
        distance = rng.uniform(10.0, 100.0)
        x = rng.uniform(0, 500)
        y = rng.uniform(0, 500)
        devices[i] = Device(
            device_id=i,
            compute_power=float(rng.uniform(0.5, 2.0)),
            bandwidth_mbps=float(rng.uniform(1.0, 10.0)),
            distance_m=float(distance),
            channel_quality=float(1.0 / distance),
            clustering_coefficient=float(rng.uniform(0.3, 1.0)),
            x=float(x),
            y=float(y),
        )
    return devices
