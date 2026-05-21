"""
devices.py — IoT device model for UAV-assisted HFL.

Changes from the previous version:
  • REMOVED: clustering_coefficient (was a random uniform scalar — abuse of
    terminology). The real graph CC is now computed by topology.compute_metrics.
  • ADDED:   `position: (x, y)` — 2-D coordinates in a bounded area, used to
    construct the actual communication graph.
  • ADDED:   `bandwidth_baseline` — the bandwidth around which round-by-round
    fluctuation happens (so noise doesn't compound).
  • ADDED:   `is_active` — set to False on round-by-round transient dropouts.
  • ADDED:   `compute_device_scores()` — used by clustering.py for head selection.

`distance` is retained but reinterpreted: distance from the UAV (not relevant
to D2D edges, which use Euclidean distance between positions).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional

import numpy as np


@dataclass
class IoTDevice:
    device_id: int

    # ── Physical capability ─────────────────────────────────────────────────
    compute_power: float          # FLOPS scaling [0.5, 2.0]
    bandwidth: float              # Mbps (current, after fluctuation)
    bandwidth_baseline: float     # Mbps (mean / nominal)
    distance: float               # metres, from the UAV  (not D2D)

    # ── Spatial position (for D2D graph) ────────────────────────────────────
    position: Tuple[float, float] # (x, y) metres in [0, area_size]²

    # ── Dynamic state ───────────────────────────────────────────────────────
    is_active: bool = True
    score: float = 0.0
    cluster_id: int = -1
    is_cluster_head: bool = False

    # Legacy / unused (kept for back-compat with code paths we haven't touched)
    channel_quality: float = 0.0
    residual_buffer: Optional[Any] = None
    energy_used: float = 0.0

    # ── Latency helpers (unchanged) ─────────────────────────────────────────
    def compute_time(self, base_compute_time: float) -> float:
        return base_compute_time / max(self.compute_power, 1e-6)

    def comm_time(self, message_size_mb: float) -> float:
        return (message_size_mb * 8.0) / max(self.bandwidth, 1e-6)

    def total_time(self, base_compute_time: float, message_size_mb: float) -> float:
        return self.compute_time(base_compute_time) + self.comm_time(message_size_mb)


def compute_device_scores(
    devices: List["IoTDevice"],
    w_compute: float,
    w_clustering: float,
    w_bandwidth: float,
) -> None:
    """
    Compute a composite score for each device and write it to device.score.

    Score = w_compute  * normalised(compute_power)
           + w_clustering * normalised(1 / distance_to_UAV)   [proxy for centrality]
           + w_bandwidth  * normalised(bandwidth)

    Scores are written in-place; the function is intentionally side-effecting
    so that select_cluster_heads() can sort by device.score immediately after.
    All three components are min-max normalised to [0, 1] across the population
    to make the weights dimensionally consistent.
    """
    if not devices:
        return

    def _norm(vals: List[float]) -> List[float]:
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        if rng < 1e-12:
            return [0.5] * len(vals)
        return [(v - lo) / rng for v in vals]

    compute_vals = [d.compute_power for d in devices]
    # Centrality proxy: closer to UAV (smaller distance) → higher score.
    # Use 1/(distance+1) to avoid division by zero and keep ordering intuitive.
    centrality_vals = [1.0 / (d.distance + 1.0) for d in devices]
    bw_vals = [d.bandwidth for d in devices]

    compute_n   = _norm(compute_vals)
    centrality_n = _norm(centrality_vals)
    bw_n        = _norm(bw_vals)

    for i, d in enumerate(devices):
        d.score = (
            w_compute   * compute_n[i]
            + w_clustering * centrality_n[i]
            + w_bandwidth  * bw_n[i]
        )


def create_devices(num_devices: int, seed: int,
                   area_size: float = 500.0) -> List[IoTDevice]:
    """
    Deterministic heterogeneous device population on a square area.
    `area_size` is the side length in metres (positions ∈ [0, area_size]²).
    """
    rng = np.random.RandomState(seed)
    devices: List[IoTDevice] = []
    for i in range(num_devices):
        bw_baseline = float(rng.uniform(1.0, 10.0))
        pos = (float(rng.uniform(0.0, area_size)),
               float(rng.uniform(0.0, area_size)))
        # distance from UAV: assume UAV hovers at centre of the area
        ux, uy = area_size / 2.0, area_size / 2.0
        dist_to_uav = float(np.hypot(pos[0] - ux, pos[1] - uy))

        devices.append(IoTDevice(
            device_id=i,
            compute_power=float(rng.uniform(0.5, 2.0)),
            bandwidth=bw_baseline,
            bandwidth_baseline=bw_baseline,
            distance=dist_to_uav,
            position=pos,
            channel_quality=1.0 / max(dist_to_uav, 1.0),
        ))
    return devices
