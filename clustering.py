from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass
class Cluster:
    cluster_id: int
    devices: List

    def aggregate(self, updates: List[torch.Tensor]) -> torch.Tensor:
        if len(updates) == 0:
            raise ValueError("No updates to aggregate.")
        return torch.stack(updates, dim=0).mean(dim=0)

    def select_head(self, devices: List | None = None):
        target_devices = self.devices if devices is None else devices
        return select_cluster_head(target_devices)


def select_cluster_head(cluster: Sequence):
    if len(cluster) == 0:
        raise ValueError("Cluster has no devices.")

    def score(device) -> float:
        compute_power = float(getattr(device, "compute_power", 0.0))
        clustering_coeff = float(getattr(device, "clustering_coeff", 0.0))
        bandwidth = float(getattr(device, "bandwidth", getattr(device, "bandwidth_mbps", 0.0)))
        return 0.5 * compute_power + 0.3 * clustering_coeff + 0.2 * bandwidth

    return max(cluster, key=score)


def cluster_devices(devices: Sequence, num_clusters: int, seed: int) -> Tuple[List[Cluster], Dict[int, object]]:
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if len(devices) == 0:
        return [], {}

    indexed = list(devices)
    indexed.sort(key=lambda d: int(getattr(d, "device_id", 0)))

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(indexed), generator=gen).tolist()
    ordered = [indexed[i] for i in perm]

    buckets: List[List] = [[] for _ in range(num_clusters)]
    for idx, device in enumerate(ordered):
        buckets[idx % num_clusters].append(device)

    clusters = [Cluster(cluster_id=i, devices=buckets[i]) for i in range(num_clusters)]

    cluster_heads: Dict[int, object] = {}
    for cluster in clusters:
        if cluster.devices:
            cluster_heads[cluster.cluster_id] = select_cluster_head(cluster.devices)

    return clusters, cluster_heads
