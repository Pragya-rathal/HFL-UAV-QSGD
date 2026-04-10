"""
Clustering module: cluster-head selection + distance-aware cluster formation.
Deterministic per seed – identical clusters used across all methods.
"""

import numpy as np
from typing import List, Dict, Tuple
from devices import IoTDevice, compute_device_scores


def select_cluster_heads(
    devices: List[IoTDevice],
    num_clusters: int,
    w_compute: float,
    w_clustering: float,
    w_bandwidth: float,
) -> List[int]:
    """Return indices of the top-scoring cluster-head candidates."""
    compute_device_scores(devices, w_compute, w_clustering, w_bandwidth)
    sorted_ids = sorted(range(len(devices)), key=lambda i: devices[i].score, reverse=True)
    head_ids = sorted_ids[:num_clusters]
    for dev in devices:
        dev.is_cluster_head = False
    for idx in head_ids:
        devices[idx].is_cluster_head = True
    return head_ids


def form_clusters(
    devices: List[IoTDevice],
    head_ids: List[int],
    max_cluster_size: int,
) -> Dict[int, List[int]]:
    """
    Assign each non-head device to the nearest cluster head (by Euclidean
    distance in a synthetic 2-D layout seeded from device distances).
    Returns {head_id: [member_device_ids]}.
    """
    # Build a synthetic 2-D position for each device using its distance as
    # the radial coordinate (angle spread uniformly around origin to maintain
    # determinism without an extra seed).
    n = len(devices)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.stack(
        [np.array([d.distance * np.cos(angles[i]),
                   d.distance * np.sin(angles[i])]) for i, d in enumerate(devices)]
    )  # shape (n, 2)

    head_positions = positions[head_ids]   # (K, 2)

    clusters: Dict[int, List[int]] = {h: [h] for h in head_ids}
    cluster_counts = {h: 1 for h in head_ids}

    non_heads = [i for i in range(n) if i not in head_ids]
    # Sort non-heads by their best distance to any head (closest first → fairer)
    dists_to_best = [
        np.min(np.linalg.norm(head_positions - positions[i], axis=1))
        for i in non_heads
    ]
    order = np.argsort(dists_to_best)

    for idx in order:
        dev_id = non_heads[idx]
        dists = np.linalg.norm(head_positions - positions[dev_id], axis=1)
        # Try heads in ascending distance order, respect max_cluster_size
        for head_rank in np.argsort(dists):
            h = head_ids[head_rank]
            if cluster_counts[h] < max_cluster_size:
                clusters[h].append(dev_id)
                cluster_counts[h] += 1
                break
        else:
            # If all clusters full, assign to nearest regardless
            h = head_ids[int(np.argmin(dists))]
            clusters[h].append(dev_id)
            cluster_counts[h] += 1

    # Write cluster_id back to device objects
    for h, members in clusters.items():
        for dev_id in members:
            devices[dev_id].cluster_id = h

    return clusters


def build_clustering(
    devices: List[IoTDevice],
    num_clusters: int,
    cfg,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Top-level convenience: select heads + form clusters.
    Returns (head_ids, clusters).
    """
    head_ids = select_cluster_heads(
        devices,
        num_clusters,
        cfg.score_w_compute,
        cfg.score_w_clustering,
        cfg.score_w_bandwidth,
    )
    clusters = form_clusters(devices, head_ids, cfg.max_cluster_size)
    return head_ids, clusters
