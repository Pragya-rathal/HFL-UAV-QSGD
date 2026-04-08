from typing import Dict, List, Tuple

import numpy as np

from devices import DeviceProfile


def form_clusters(
    devices: Dict[int, DeviceProfile],
    num_clusters: int,
    method: str,
    seed: int,
) -> Dict[int, List[int]]:
    ids = list(devices.keys())
    rng = np.random.default_rng(seed)

    if method == "random":
        rng.shuffle(ids)
    elif method == "distance":
        ids = sorted(ids, key=lambda d: devices[d].distance_m)
    else:
        raise ValueError(f"Unknown cluster formation method: {method}")

    clusters = {c: [] for c in range(num_clusters)}
    for idx, device_id in enumerate(ids):
        clusters[idx % num_clusters].append(device_id)
    return clusters


def choose_cluster_heads(
    clusters: Dict[int, List[int]],
    devices: Dict[int, DeviceProfile],
    weights: Tuple[float, float, float],
) -> Dict[int, int]:
    a, b, c = weights
    heads = {}
    for cid, members in clusters.items():
        best = max(
            members,
            key=lambda d: a * devices[d].compute_power
            + b * devices[d].clustering_coefficient
            + c * devices[d].bandwidth_mbps,
        )
        heads[cid] = best
    return heads


def select_quorum(
    members: List[int],
    devices: Dict[int, DeviceProfile],
    quorum_fraction: float,
    quorum_count: int,
    score_weights: Tuple[float, float],
    jitter: float,
    rng: np.random.Generator,
) -> List[int]:
    w1, w2 = score_weights
    if quorum_count > 0:
        q = min(quorum_count, len(members))
    else:
        q = max(1, int(np.ceil(quorum_fraction * len(members))))

    scored = []
    for d in members:
        s = w1 * devices[d].compute_power + w2 * devices[d].bandwidth_mbps
        s += rng.normal(0.0, jitter)
        scored.append((s, d))
    scored.sort(reverse=True)
    return [d for _, d in scored[:q]]
