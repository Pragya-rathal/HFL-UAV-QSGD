"""
main.py — HFL-UAV / TopoCo orchestration.

CLI:
    python main.py --mode toy
    python main.py --mode full
    python main.py --mode toy --only-main           # skip ablations
    python main.py --mode full --static-compression # ablation toggle (from config)

Notebook:
    from main import run_from_notebook
    run_from_notebook(mode="full")

Method registry
───────────────
  standard_fl       FedAvg, no clustering
  clustered_fl      FedAvg with clustering
  topk_ef           Cluster + Top-K + EF
  qsgd              Cluster + QSGD
  topk_quorum       Cluster + Top-K + heuristic top-K participation  (legacy)
  qsgd_quorum       Cluster + QSGD     + heuristic top-K participation  (legacy)
  fedprox           HFL + FedProx (modern baseline)
  topoco            HFL + primal-dual topology-adaptive co-optimisation (OURS)
"""

import os
import copy
import time
from typing import Dict

import numpy as np
import pandas as pd
import torch

from config import parse_args, Config
from data_loader import load_data
from model import get_model, count_parameters
from devices import create_devices
from clustering import build_initial_clustering
from topology import build_graph
from federated import run_method
from metrics import (
    history_to_df, aggregate_seeds, compute_summary,
    print_summary_table, get_cluster_latency_stats,
)
from plotting import generate_all_plots
from plotting_topoco import generate_topoco_plots


# ─────────────────────────────────────────────────────────────────────────────
# Method registry
# ─────────────────────────────────────────────────────────────────────────────

CLASSIC_METHODS = [
    "standard_fl", "clustered_fl", "topk_ef", "qsgd",
    "topk_quorum", "qsgd_quorum",
]
MODERN_BASELINES = ["fedprox"]
OURS = ["topoco"]
EXPERIMENT_METHODS = CLASSIC_METHODS + MODERN_BASELINES + OURS


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility / CUDA
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_cuda(cfg: Config):
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] cfg.device='cuda' but CUDA unavailable → CPU.")
            cfg.device = "cpu"; return
        torch.backends.cudnn.benchmark = True
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"[INFO] CUDA: {p.name}  |  {p.total_memory/1e9:.1f} GB")
    else:
        print("[INFO] CPU.")


# ─────────────────────────────────────────────────────────────────────────────
# Single run — build the per-seed environment once, then call run_method
# ─────────────────────────────────────────────────────────────────────────────

def _build_environment(seed: int, cfg: Config):
    """Returns (train_loaders, test_loader, devices, head_ids, clusters)."""
    set_seed(seed)
    train_loaders, test_loader = load_data(
        cfg.dataset, cfg.num_devices, cfg.iid, cfg.alpha,
        seed, cfg.batch_size, cfg.test_batch_size, device=cfg.device,
    )
    devices = create_devices(cfg.num_devices, seed, area_size=cfg.area_size)
    G0 = build_graph(devices, cfg.r_comm)
    head_ids, clusters = build_initial_clustering(devices, G0, cfg)
    return train_loaders, test_loader, devices, head_ids, clusters


def run_single(method: str, seed: int, cfg: Config):
    print(f"\n{'='*64}\n  {method}  |  seed={seed}  |  mode={cfg.mode}  |  dev={cfg.device}\n{'='*64}")
    train_loaders, test_loader, devices, head_ids, clusters = _build_environment(seed, cfg)

    set_seed(seed)
    global_model_init = get_model(cfg.dataset, cfg.device)
    print(f"  model params: {count_parameters(global_model_init):,}")

    # Store the current seed on cfg so that run_topoco_method (and any future
    # method wrappers) can derive a correctly seeded internal RNG, ensuring
    # full reproducibility across seeds.
    cfg._current_seed = seed

    history = run_method(method, global_model_init, train_loaders, test_loader,
                         devices, clusters, head_ids, cfg)

    del train_loaders, test_loader, global_model_init
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return history


def run_experiment(cfg: Config) -> dict:
    """Returns {method → [per-seed history]}."""
    results = {m: [] for m in EXPERIMENT_METHODS}
    for seed in cfg.seeds:
        for method in EXPERIMENT_METHODS:
            results[method].append(run_single(method, seed, cfg))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Adaptivity ablation: static / adaptive × compression / participation
# ─────────────────────────────────────────────────────────────────────────────

def run_adaptivity_ablation(cfg: Config) -> Dict[str, float]:
    """
    Four variants of `topoco`:
      (1) static comp / static part   ←  baseline (essentially clustered+topK)
      (2) adaptive comp / static part
      (3) static comp / adaptive part
      (4) adaptive comp / adaptive part   ←  full TopoCo
    Reports best accuracy for each (one seed for speed; multi-seed in main exp).
    """
    seed = cfg.seeds[0]
    out: Dict[str, float] = {}

    for label, ac, ap in [
        ("static_C / static_P",   False, False),
        ("adaptive_C / static_P", True,  False),
        ("static_C / adaptive_P", False, True),
        ("adaptive_C / adaptive_P (full)", True, True),
    ]:
        print(f"\n[Ablation: adaptivity] {label}")
        c2 = copy.copy(cfg)
        c2.adaptive_compression_enabled  = ac
        c2.adaptive_participation_enabled = ap
        c2.seeds = [seed]
        hist = run_single("topoco", seed, c2)
        out[label] = max(h["accuracy"] for h in hist)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, cfg: Config, mode_dir: str):
    all_dfs, agg_dfs = {}, {}
    for method, seed_histories in results.items():
        method_dir = os.path.join(mode_dir, method)
        os.makedirs(os.path.join(method_dir, "plots"), exist_ok=True)
        seed_dfs = []
        for seed, history in zip(cfg.seeds, seed_histories):
            df = history_to_df(history, method, seed)
            df.to_csv(os.path.join(method_dir, f"metrics_seed{seed}.csv"), index=False)
            seed_dfs.append(df)
        pd.concat(seed_dfs).to_csv(os.path.join(method_dir, "metrics.csv"), index=False)
        all_dfs[method] = seed_dfs
        agg_dfs[method] = aggregate_seeds(seed_dfs)
    summary_df = compute_summary(all_dfs)
    return all_dfs, agg_dfs, summary_df


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: Config):
    setup_cuda(cfg)
    mode_dir      = os.path.join(cfg.results_dir, cfg.mode)
    summaries_dir = os.path.join(cfg.results_dir, "summaries")
    plots_dir     = os.path.join(mode_dir, "plots")
    for d in (mode_dir, summaries_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\n{'#'*64}")
    print(f"  HFL-UAV / TopoCo  |  Mode: {cfg.mode.upper()}")
    print(f"  Dataset : {cfg.dataset}  |  Devices: {cfg.num_devices}  |  Clusters: {cfg.num_clusters}")
    print(f"  Rounds  : {cfg.num_rounds}  |  Seeds: {cfg.seeds}  |  IID: {cfg.iid}  |  α: {cfg.alpha}")
    print(f"  Methods : {EXPERIMENT_METHODS}")
    print(f"  Device  : {cfg.device}")
    print(f"{'#'*64}\n")

    # ── Main experiment ─────────────────────────────────────────────────────
    t0 = time.time()
    results = run_experiment(cfg)
    print(f"\n[INFO] Main experiment: {(time.time()-t0)/60:.1f} min.")

    all_dfs, agg_dfs, summary_df = save_results(results, cfg, mode_dir)
    summary_df.to_csv(os.path.join(summaries_dir, f"{cfg.mode}_summary.csv"), index=False)
    print_summary_table(summary_df)

    # ── Ablation: adaptivity (the headline ablation for the paper) ──────────
    t = time.time()
    print("\n[INFO] Running adaptivity ablation…")
    ablation_results = run_adaptivity_ablation(cfg)
    print(f"[INFO] Ablation: {(time.time()-t)/60:.1f} min.")
    for k, v in ablation_results.items():
        print(f"    {k:40s}  best_acc = {v:.4f}")

    # ── Plots ───────────────────────────────────────────────────────────────
    cluster_lat_dfs = {m: get_cluster_latency_stats(s) for m, s in results.items()}
    print("\n[INFO] Generating base plots…")
    n_base = generate_all_plots(agg_dfs, summary_df, cluster_lat_dfs, plots_dir)
    print("[INFO] Generating TopoCo plots…")
    n_topoco = generate_topoco_plots(all_dfs, summary_df, ablation_results, plots_dir)
    print(f"[INFO] Plots written: {n_base} base + {n_topoco} TopoCo")

    summary_df.to_csv(os.path.join(summaries_dir, "aggregate.csv"), index=False)

    # Persist ablation CSV
    pd.DataFrame([
        {"variant": k, "best_acc": v} for k, v in ablation_results.items()
    ]).to_csv(os.path.join(summaries_dir, f"{cfg.mode}_ablation_adaptivity.csv"), index=False)

    print(f"\n{'='*64}\n  Done.  Results: {os.path.abspath(mode_dir)}\n{'='*64}\n")
    return results, summary_df, ablation_results


def run_from_notebook(mode="toy", iid=False, alpha=0.5, device=None,
                      results_dir="results", seeds=None):
    cfg = Config()
    cfg.mode = mode; cfg.iid = iid; cfg.alpha = alpha
    cfg.results_dir = results_dir
    if device is not None: cfg.device = device
    cfg.apply_mode()
    if seeds is not None: cfg.seeds = seeds
    return run(cfg)


def main():
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
