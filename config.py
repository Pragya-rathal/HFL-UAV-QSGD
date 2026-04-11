"""
Configuration for Hierarchical Federated Learning in UAV-assisted IoT Networks.
IEEE Transactions-level study.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ─── Mode ────────────────────────────────────────────────────────────────
    mode: str = "toy"           # "toy" | "full"

    # ─── Dataset ─────────────────────────────────────────────────────────────
    dataset: str = "MNIST"      # auto-set from mode
    iid: bool = False           # IID vs Non-IID
    alpha: float = 0.5          # Dirichlet concentration (non-IID)
    batch_size: int = 32
    test_batch_size: int = 256

    # ─── Devices ─────────────────────────────────────────────────────────────
    num_devices: int = 25
    num_clusters: int = 5

    # ─── Training ────────────────────────────────────────────────────────────
    num_rounds: int = 22
    local_epochs: int = 2
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # ─── Clustering ──────────────────────────────────────────────────────────
    score_w_compute: float = 0.5
    score_w_clustering: float = 0.3
    score_w_bandwidth: float = 0.2
    max_cluster_size: int = 10

    # ─── Compression ─────────────────────────────────────────────────────────
    topk_fraction: float = 0.1     # fraction of params to keep
    qsgd_levels: int = 8           # quantization levels

    # ─── Quorum ───────────────────────────────────────────────────────────────
    quorum_fraction: float = 0.6   # fraction of cluster to select

    # ─── Physical / Latency ──────────────────────────────────────────────────
    base_compute_time: float = 1.0   # seconds per unit workload
    model_bits: int = 32
    agg_head_time: float = 0.05     # seconds
    uav_comm_base: float = 0.1      # seconds

    # ─── Seeds ───────────────────────────────────────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [42, 7, 123])

    # ─── Output ──────────────────────────────────────────────────────────────
    results_dir: str = "results"
    device: str = "cpu"             # auto-detected

    def apply_mode(self):
        if self.mode == "toy":
            self.dataset = "MNIST"
            self.num_devices = 25
            self.num_clusters = 5
            self.num_rounds = 22
            self.local_epochs = 2
            self.seeds = [42, 7, 123]
        else:  # full
            self.dataset = "CIFAR10"
            self.num_devices = 60
            self.num_clusters = 10
            self.num_rounds = 60
            self.local_epochs = 4
            self.seeds = [42, 7, 123, 17, 99]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="HFL-UAV Federated Learning Study")
    parser.add_argument("--mode", choices=["toy", "full"], default="toy")
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = Config()
    cfg.mode = args.mode
    cfg.iid = args.iid
    cfg.alpha = args.alpha
    cfg.device = args.device
    cfg.apply_mode()
    return cfg
