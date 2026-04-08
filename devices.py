from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class DeviceProfile:
    device_id: int
    compute_power: float
    bandwidth_mbps: float
    distance_m: float
    channel_quality: float
    clustering_coefficient: float
    residual: Dict[str, torch.Tensor] = field(default_factory=dict)

    def init_residual(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if not self.residual:
            self.residual = {k: torch.zeros_like(v) for k, v in state_dict.items()}


def get_model_delta(local_model: nn.Module, global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    local_state = local_model.state_dict()
    return {k: local_state[k] - global_state[k] for k in global_state}


def apply_delta(state: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor], weight: float = 1.0) -> Dict[str, torch.Tensor]:
    return {k: state[k] + weight * delta[k] for k in state}


def weighted_average_deltas(deltas, weights):
    total = float(sum(weights))
    avg = {k: torch.zeros_like(deltas[0][k]) for k in deltas[0]}
    for d, w in zip(deltas, weights):
        for k in avg:
            avg[k] += d[k] * (w / total)
    return avg
