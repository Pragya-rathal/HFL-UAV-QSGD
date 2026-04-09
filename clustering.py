from dataclasses import dataclass
from typing import List

import torch

from devices import Device


@dataclass
class Cluster:
    cluster_id: int

    def aggregate(self, updates: List[torch.Tensor]) -> torch.Tensor:
        if len(updates) == 0:
            raise ValueError("No updates to aggregate.")
        stacked = torch.stack(updates, dim=0)
        return stacked.mean(dim=0)

    def select_head(self, devices: List[Device]) -> Device:
        if len(devices) == 0:
            raise ValueError("Cluster has no devices.")
        return max(devices, key=lambda d: d.bandwidth_mbps * d.compute_factor)
