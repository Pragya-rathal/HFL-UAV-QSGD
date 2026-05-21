"""
Configuration — extended with topology, dynamics, Lagrangian, and schedule
parameters for the topology-adaptive co-optimization framework.

All new fields have sensible defaults; running without setting them reproduces
the previous static framework as a special case (set lambda step sizes to 0,
dropout_prob to 0, q_min = q_max = topk_fraction).
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # ─── Mode ────────────────────────────────────────────────────────────────
    mode: str = "toy"
    dataset: str = "MNIST"
    iid: bool = False
    alpha: float = 0.5
    batch_size: int = 32
    test_batch_size: int = 256

    # ─── Devices ─────────────────────────────────────────────────────────────
    num_devices: int = 25
    num_clusters: int = 5
    area_size: float = 500.0          # metres, side length of square area
    r_comm: float = 150.0             # metres, D2D communication radius

    # ─── Training ────────────────────────────────────────────────────────────
    num_rounds: int = 22
    local_epochs: int = 2
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # ─── Topology evolution ─────────────────────────────────────────────────
    bw_noise_std:   float = 0.10      # multiplicative bandwidth noise std
    dropout_prob:   float = 0.02      # per-round per-device transient dropout
    mobility_step:  float = 0.0       # metres of jitter per round (0 = static positions)
    K_topo:         int   = 5         # recompute APL/betweenness every K_topo rounds

    # ─── Clustering ──────────────────────────────────────────────────────────
    max_cluster_size:   int = 10
    rehead_every:       int = 10      # re-elect cluster heads every N rounds (0 = never)
    initial_score_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
        # initial (w_compute, w_topology, w_bandwidth) — used only at t=0 to
        # bootstrap clusters before the Lagrangian utility takes over

    # ─── Compression ─────────────────────────────────────────────────────────
    topk_fraction: float = 0.1        # used by NON-adaptive baselines
    qsgd_levels:   int   = 8
    adaptive_compression_enabled: bool = True

    # ─── Legacy field — used only by the original topk_quorum/qsgd_quorum
    #     methods that survive as baselines. New TopoCo path ignores this. ──
    quorum_fraction: float = 0.6

    # ─── Schedule (adaptive primal policies) ────────────────────────────────
    q_min:    float = 0.02
    q_max:    float = 0.50
    rho_max:  float = 1.0             # max participation fraction per cluster
    floor_m:  int   = 2               # min participants per cluster
    adaptive_participation_enabled: bool = True

    # ─── Lagrangian / dual variables ────────────────────────────────────────
    lambda_C_init: float = 0.5
    lambda_L_init: float = 0.5
    lambda_D_init: float = 0.5
    eta_C: float = 0.10
    eta_L: float = 0.10
    eta_D: float = 0.10
    lambda_max: float = 10.0
    gamma_T:    float = 1.0           # weight on topology utility T_i in score

    # Constraint targets (per round)
    B_target_mb: float = 50.0
    L_target_s: float  = 5.0
    D_target:   float  = 1.0

    # ─── Physical / Latency (used for predictions + simulation) ─────────────
    base_compute_time: float = 1.0
    model_bits:        int   = 32
    agg_head_time:     float = 0.05
    uav_comm_base:     float = 0.1

    # ─── Seeds ───────────────────────────────────────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [42, 7, 123])

    # ─── Output / Device ─────────────────────────────────────────────────────
    results_dir: str = "results"
    device: str = field(default_factory=_default_device)

    # ─── Mode preset ────────────────────────────────────────────────────────
    def apply_mode(self):
        if self.mode == "toy":
            self.dataset = "MNIST"
            self.num_devices = 25
            self.num_clusters = 5
            self.num_rounds = 22
            self.local_epochs = 2
            self.seeds = [42, 7, 123]
            self.area_size = 500.0
            self.r_comm = 150.0
            self.B_target_mb = 20.0
        else:  # full
            self.dataset = "CIFAR-10"
            self.num_devices = 60
            self.num_clusters = 10
            self.num_rounds = 60
            self.local_epochs = 4
            self.seeds = [42, 7, 123, 17, 99]
            self.area_size = 1000.0
            self.r_comm = 250.0
            self.B_target_mb = 80.0


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="HFL-UAV — topology-adaptive co-optimization")
    parser.add_argument("--mode", choices=["toy", "full"], default="toy")
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)

    # Toggle adaptivity for ablations
    parser.add_argument("--static-compression", action="store_true",
                        help="disable adaptive compression (use topk_fraction)")
    parser.add_argument("--static-participation", action="store_true",
                        help="disable adaptive participation (use rho_max for all)")
    parser.add_argument("--mobility", type=float, default=None,
                        help="metres of position jitter per round")

    args = parser.parse_args()
    cfg = Config()
    cfg.mode = args.mode; cfg.iid = args.iid; cfg.alpha = args.alpha
    if args.device is not None: cfg.device = args.device
    cfg.apply_mode()
    if args.seeds is not None: cfg.seeds = args.seeds
    if args.static_compression:   cfg.adaptive_compression_enabled = False
    if args.static_participation: cfg.adaptive_participation_enabled = False
    if args.mobility is not None: cfg.mobility_step = float(args.mobility)
    return cfg
