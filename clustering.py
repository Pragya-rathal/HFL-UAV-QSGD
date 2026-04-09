from typing import Dict, List, Tuple

import numpy as np

from devices import Device



def select_cluster_heads(
    devices: Dict[int, Device],
    num_clusters: int,
    weights: Dict[str, float],
) -> List[int]:
    scored = []
    for d in devices.values():
        score = (
            weights["compute"] * d.compute_power
            + weights["clustering"] * d.clustering_coefficient
            + weights["bandwidth"] * (d.bandwidth_mbps / 10.0)
        )
        scored.append((d.device_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:num_clusters]]


def make_clusters(
    devices: Dict[int, Device],
    heads: List[int],
) -> Dict[int, List[int]]:
    clusters = {h: [h] for h in heads}
    for device_id, device in devices.items():
        if device_id in heads:
            continue
        dists = []
        for h in heads:
            hd = devices[h]
            eu = np.sqrt((device.x - hd.x) ** 2 + (device.y - hd.y) ** 2)
            dists.append((h, eu))
        nearest = min(dists, key=lambda x: x[1])[0]
        clusters[nearest].append(device_id)
    return clusters


def standard_topology(devices: Dict[int, Device]) -> Tuple[List[int], Dict[int, List[int]]]:
    heads = list(devices.keys())
    clusters = {d: [d] for d in devices}
    return heads, clusters
