"""
Configuration for Hierarchical Federated Learning in UAV-assisted IoT Networks.
"""

import argparse
from dataclasses import dataclass, field
from typing import List

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_ablation_methods() -> List[str]:
    # The main comparison table (in run_experiment) uses all 6 methods.
    # Ablations only need the baselines + proposed methods — re-running C/D
    # in the ablations just re-derives behaviour already pinned by the main run.
    return ["standard_fl", "clustered_fl", "topk_quorum", "qsgd_quorum"]


@dataclass
class Config:
    # ─── Mode ────────────────────────────────────────────────────────────────
    mode: str = "toy"               # "toy" | "full"

    # ─── Dataset ─────────────────────────────────────────────────────────────
    dataset: str = "MNIST"
    iid: bool = False
    alpha: float = 0.5
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
    topk_fraction: float = 0.1
    qsgd_levels: int = 8

    # ─── Quorum ──────────────────────────────────────────────────────────────
    quorum_fraction: float = 0.6

    # ─── Physical / Latency ──────────────────────────────────────────────────
    base_compute_time: float = 1.0
    model_bits: int = 32
    agg_head_time: float = 0.05
    uav_comm_base: float = 0.1

    # ─── Seeds ───────────────────────────────────────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [42, 7, 123])

    # ─── Output / Device ─────────────────────────────────────────────────────
    results_dir: str = "results"
    device: str = field(default_factory=_default_device)

    # ─── Experiment block toggles ────────────────────────────────────────────
    # Main experiment always runs. These gate only the ablations.
    run_quorum:    bool = True      # quorum_fraction sweep
    run_scaling:   bool = True      # num_devices sweep
    run_alpha:     bool = True      # Dirichlet α sweep
    run_bandwidth: bool = False     # bandwidth-scale sweep — OFF by default

    # Methods used inside scaling/alpha/bandwidth ablations.
    # Main experiment ignores this and always uses all six.
    ablation_methods: List[str] = field(default_factory=_default_ablation_methods)

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
    parser.add_argument("--mode",   choices=["toy", "full"], default="toy")
    parser.add_argument("--iid",    action="store_true")
    parser.add_argument("--alpha",  type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda | cpu (default: auto-detect)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="override seeds, e.g. --seeds 42  (default: mode preset)")

    # Ablation block toggles
    parser.add_argument("--no-quorum",   dest="run_quorum",    action="store_false")
    parser.add_argument("--no-scaling",  dest="run_scaling",   action="store_false")
    parser.add_argument("--no-alpha",    dest="run_alpha",     action="store_false")
    parser.add_argument("--bandwidth",   dest="run_bandwidth", action="store_true",
                        help="enable the bandwidth-scale robustness sweep (default: off)")
    parser.add_argument("--only-main",   action="store_true",
                        help="shortcut: disable all ablations")

    args = parser.parse_args()

    cfg = Config()
    cfg.mode  = args.mode
    cfg.iid   = args.iid
    cfg.alpha = args.alpha
    if args.device is not None:
        cfg.device = args.device

    cfg.apply_mode()
    if args.seeds is not None:
        cfg.seeds = args.seeds

    cfg.run_quorum    = args.run_quorum
    cfg.run_scaling   = args.run_scaling
    cfg.run_alpha     = args.run_alpha
    cfg.run_bandwidth = args.run_bandwidth
    if args.only_main:
        cfg.run_quorum = cfg.run_scaling = cfg.run_alpha = cfg.run_bandwidth = False

    return cfg
