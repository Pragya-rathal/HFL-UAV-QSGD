"""
Clustering module: cluster-head selection + distance-aware cluster formation.
Deterministic per seed – identical clusters used across all methods.

Public API
──────────
  build_initial_clustering(devices, G, cfg)  → (head_ids, clusters)
      Top-level entry called by main.py.  Wraps build_clustering and uses
      the graph G to break ties via degree (richer node = better head).

  build_clustering(devices, num_clusters, cfg) → (head_ids, clusters)
      Legacy convenience used by some internal paths.

  reelect_heads(devices, G, scores, cfg)    → (head_ids, clusters)
      Called every cfg.rehead_every rounds inside TopoCo to refresh heads
      based on the current primal-dual scores.

  select_cluster_heads(devices, num_clusters, w_compute, w_clustering, w_bandwidth)
      → List[int]  (head device indices)

  form_clusters(devices, head_ids, max_cluster_size)
      → Dict[int, List[int]]
"""

import numpy as np
from typing import List, Dict, Tuple
import networkx as nx

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
    dists_to_best = [
        np.min(np.linalg.norm(head_positions - positions[i], axis=1))
        for i in non_heads
    ]
    order = np.argsort(dists_to_best)

    for idx in order:
        dev_id = non_heads[idx]
        dists = np.linalg.norm(head_positions - positions[dev_id], axis=1)
        for head_rank in np.argsort(dists):
            h = head_ids[head_rank]
            if cluster_counts[h] < max_cluster_size:
                clusters[h].append(dev_id)
                cluster_counts[h] += 1
                break
        else:
            h = head_ids[int(np.argmin(dists))]
            clusters[h].append(dev_id)
            cluster_counts[h] += 1

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
    Reads weight triple from cfg.initial_score_weights.
    Returns (head_ids, clusters).
    """
    w_compute, w_clustering, w_bandwidth = cfg.initial_score_weights
    head_ids = select_cluster_heads(
        devices,
        num_clusters,
        w_compute,
        w_clustering,
        w_bandwidth,
    )
    clusters = form_clusters(devices, head_ids, cfg.max_cluster_size)
    return head_ids, clusters


def build_initial_clustering(
    devices: List[IoTDevice],
    G,                          # nx.Graph built by topology.build_graph
    cfg,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Entry point called by main.py at the start of each seed run.

    Uses cfg.initial_score_weights for the initial head-selection score and
    additionally breaks ties by graph degree: among equally-scored candidates,
    a node with more neighbours in G makes a better cluster head because it
    can directly reach more members.

    Falls back gracefully when G has no edges (isolated devices).
    Returns (head_ids, clusters).
    """
    w_compute, w_clustering, w_bandwidth = cfg.initial_score_weights
    compute_device_scores(devices, w_compute, w_clustering, w_bandwidth)

    # Add a small degree-based tiebreaker (normalised to [0, 0.05])
    if G is not None and G.number_of_edges() > 0:
        max_deg = max(dict(G.degree()).values()) + 1e-9
        for d in devices:
            deg_bonus = 0.05 * (G.degree(d.device_id) / max_deg
                                if d.device_id in G else 0.0)
            d.score += deg_bonus

    sorted_ids = sorted(range(len(devices)), key=lambda i: devices[i].score, reverse=True)
    num_clusters = cfg.num_clusters
    head_ids = sorted_ids[:num_clusters]

    for dev in devices:
        dev.is_cluster_head = False
    for idx in head_ids:
        devices[idx].is_cluster_head = True

    clusters = form_clusters(devices, head_ids, cfg.max_cluster_size)
    return head_ids, clusters


def reelect_heads(
    devices: List[IoTDevice],
    G,                          # current nx.Graph
    scores: Dict[int, float],   # per-device primal-dual scores from utility.py
    cfg,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Re-elect cluster heads based on the current round's primal-dual per-device
    scores (S_i from utility.compute_per_device_score).  Only active devices
    are eligible.  Called inside run_topoco_round every cfg.rehead_every rounds.

    Returns (new_head_ids, new_clusters).
    """
    num_clusters = cfg.num_clusters

    # Only active devices can become heads; inactive get -inf score
    eligible = [
        d.device_id for d in devices
        if d.is_active and scores.get(d.device_id, float("-inf")) > float("-inf")
    ]

    if len(eligible) < num_clusters:
        # Not enough active devices — keep existing heads
        existing_heads = [d.device_id for d in devices if d.is_cluster_head]
        if len(existing_heads) >= num_clusters:
            return existing_heads[:num_clusters], form_clusters(
                devices, existing_heads[:num_clusters], cfg.max_cluster_size
            )
        # Worst case: use top eligible + fill from all devices
        fallback = sorted(range(len(devices)),
                          key=lambda i: scores.get(devices[i].device_id, 0.0),
                          reverse=True)
        head_ids = fallback[:num_clusters]
    else:
        ranked = sorted(eligible, key=lambda did: scores[did], reverse=True)
        head_ids = ranked[:num_clusters]

    for dev in devices:
        dev.is_cluster_head = False
    for did in head_ids:
        devices[did].is_cluster_head = True

    clusters = form_clusters(devices, head_ids, cfg.max_cluster_size)
    return head_ids, clusters
