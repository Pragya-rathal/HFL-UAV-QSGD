from typing import Dict, List

from devices import Device


def select_cluster_heads(devices: Dict[int, Device], num_clusters: int, a: float = 0.45, b: float = 0.4, c: float = 0.15) -> List[int]:
    scored = []
    for d in devices.values():
        score = a * d.compute_power + b * d.clustering_coefficient + c * (d.bandwidth_mbps / 10.0)
        scored.append((d.device_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [device_id for device_id, _ in scored[:num_clusters]]


def build_clusters(devices: Dict[int, Device], heads: List[int]) -> Dict[int, List[int]]:
    clusters: Dict[int, List[int]] = {head: [head] for head in heads}
    for device_id in devices:
        if device_id in heads:
            continue
        assigned = heads[device_id % len(heads)]
        clusters[assigned].append(device_id)
    return clusters
