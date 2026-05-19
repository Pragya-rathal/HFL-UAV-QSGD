"""
Main entry point for the HFL-UAV study — GPU-native.

CLI:
    python main.py --mode toy
    python main.py --mode full

Notebook:
    from main import run_from_notebook
    run_from_notebook(mode="toy")
"""

import os
import sys
import copy
import time

import numpy as np
import pandas as pd
import torch

from config import parse_args, Config
from data_loader import load_data
from model import get_model, count_parameters
from devices import create_devices
from clustering import build_clustering
from federated import run_method
from metrics import (
    history_to_df, aggregate_seeds, compute_summary,
    print_summary_table, get_cluster_latency_stats,
)
from plotting import generate_all_plots


EXPERIMENT_METHODS = [
    "standard_fl",
    "clustered_fl",
    "topk_ef",
    "qsgd",
    "topk_quorum",
    "qsgd_quorum",
]


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_cuda(cfg: Config):
    """Print device info and enable cuDNN autotune."""
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] cfg.device='cuda' but CUDA unavailable → falling back to CPU.")
            cfg.device = "cpu"
            return
        torch.backends.cudnn.benchmark = True
        gpu_id = torch.cuda.current_device()
        props  = torch.cuda.get_device_properties(gpu_id)
        print(f"[INFO] Using CUDA: [{gpu_id}] {props.name}  |  "
              f"{props.total_memory/1e9:.1f} GB  |  cuDNN benchmark ON")
    else:
        print("[INFO] Using CPU (CUDA not requested).")


# ─── Single run ──────────────────────────────────────────────────────────────

def run_single(method: str, seed: int, cfg: Config):
    print(f"\n{'='*60}")
    print(f"  Method={method}  Seed={seed}  Mode={cfg.mode}  Device={cfg.device}")
    print(f"{'='*60}")
    set_seed(seed)

    train_loaders, test_loader = load_data(
        cfg.dataset, cfg.num_devices, cfg.iid, cfg.alpha,
        seed, cfg.batch_size, cfg.test_batch_size,
        device=cfg.device,
    )

    devices = create_devices(cfg.num_devices, seed)
    head_ids, clusters = build_clustering(devices, cfg.num_clusters, cfg)

    set_seed(seed)  # reset before model init for deterministic weights
    global_model_init = get_model(cfg.dataset, cfg.device)

    history = run_method(
        method, global_model_init, train_loaders, test_loader,
        devices, clusters, head_ids, cfg,
    )

    # Free the per-seed GPU loaders + model before the next method run
    del train_loaders, test_loader, global_model_init
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    return history


def run_experiment(cfg: Config) -> dict:
    """Returns {method → [per-seed history]}."""
    results: dict = {m: [] for m in EXPERIMENT_METHODS}
    for seed in cfg.seeds:
        for method in EXPERIMENT_METHODS:
            results[method].append(run_single(method, seed, cfg))
    return results


# ─── Ablations ────────────────────────────────────────────────────────────────

def run_quorum_sensitivity(cfg: Config) -> dict:
    fractions = [0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
    quorum_results = {}

    seed = cfg.seeds[0]
    train_loaders, test_loader = load_data(
        cfg.dataset, cfg.num_devices, cfg.iid, cfg.alpha,
        seed, cfg.batch_size, cfg.test_batch_size, device=cfg.device,
    )
    devices = create_devices(cfg.num_devices, seed)
    head_ids, clusters = build_clustering(devices, cfg.num_clusters, cfg)

    for frac in fractions:
        print(f"\n[Quorum sensitivity] fraction={frac:.2f}")
        quorum_results[frac] = {}
        for method in ("topk_quorum", "qsgd_quorum"):
            cfg_copy = copy.copy(cfg)
            cfg_copy.quorum_fraction = frac
            set_seed(seed)
            global_model_init = get_model(cfg.dataset, cfg.device)
            history = run_method(
                method, global_model_init, train_loaders, test_loader,
                devices, clusters, head_ids, cfg_copy,
            )
            best_acc   = max(h["accuracy"] for h in history)
            avg_lat    = float(np.mean([h["latency_round"] for h in history]))
            total_comm = float(sum(h["comm_total_mb"] for h in history))
            quorum_results[frac][method] = (best_acc, avg_lat, total_comm)
            del global_model_init
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

    return quorum_results


def run_scaling(cfg: Config) -> dict:
    device_counts = [10, 20, 30, 40] if cfg.mode == "toy" else [20, 40, 60, 80]
    scaling_results = {}
    seed = cfg.seeds[0]
    rounds_backup = cfg.num_rounds
    cfg.num_rounds = max(5, cfg.num_rounds // 4)

    for n in device_counts:
        print(f"\n[Scaling] num_devices={n}")
        cfg_copy = copy.copy(cfg)
        cfg_copy.num_devices = n
        cfg_copy.num_clusters = max(2, n // 5)
        scaling_results[n] = {}

        train_loaders, test_loader = load_data(
            cfg_copy.dataset, n, cfg_copy.iid, cfg_copy.alpha,
            seed, cfg_copy.batch_size, cfg_copy.test_batch_size, device=cfg.device,
        )
        devices = create_devices(n, seed)
        head_ids, clusters = build_clustering(devices, cfg_copy.num_clusters, cfg_copy)

        for method in EXPERIMENT_METHODS:
            set_seed(seed)
            global_model_init = get_model(cfg_copy.dataset, cfg.device)
            history = run_method(
                method, global_model_init, train_loaders, test_loader,
                devices, clusters, head_ids, cfg_copy,
            )
            scaling_results[n][method] = max(h["accuracy"] for h in history)
            del global_model_init
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

    cfg.num_rounds = rounds_backup
    return scaling_results


def run_robustness(cfg: Config) -> dict:
    seed = cfg.seeds[0]
    rounds_backup = cfg.num_rounds
    cfg.num_rounds = max(5, cfg.num_rounds // 4)

    robustness = {"alpha": {}, "bandwidth": {}}

    for alpha in [0.1, 0.3, 0.5, 1.0, 5.0]:
        print(f"\n[Robustness] alpha={alpha}")
        cfg_copy = copy.copy(cfg)
        cfg_copy.alpha = alpha
        train_loaders, test_loader = load_data(
            cfg_copy.dataset, cfg_copy.num_devices, False, alpha,
            seed, cfg_copy.batch_size, cfg_copy.test_batch_size, device=cfg.device,
        )
        devices = create_devices(cfg_copy.num_devices, seed)
        head_ids, clusters = build_clustering(devices, cfg_copy.num_clusters, cfg_copy)
        robustness["alpha"][alpha] = {}
        for method in EXPERIMENT_METHODS:
            set_seed(seed)
            global_model_init = get_model(cfg_copy.dataset, cfg.device)
            history = run_method(
                method, global_model_init, train_loaders, test_loader,
                devices, clusters, head_ids, cfg_copy,
            )
            robustness["alpha"][alpha][method] = max(h["accuracy"] for h in history)
            del global_model_init
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

    for bw_scale in [0.5, 0.75, 1.0, 1.5, 2.0]:
        print(f"\n[Robustness] bw_scale={bw_scale}")
        devices_scaled = create_devices(cfg.num_devices, seed)
        for d in devices_scaled:
            d.bandwidth *= bw_scale
        head_ids, clusters = build_clustering(devices_scaled, cfg.num_clusters, cfg)
        train_loaders, test_loader = load_data(
            cfg.dataset, cfg.num_devices, cfg.iid, cfg.alpha,
            seed, cfg.batch_size, cfg.test_batch_size, device=cfg.device,
        )
        robustness["bandwidth"][bw_scale] = {}
        for method in EXPERIMENT_METHODS:
            set_seed(seed)
            global_model_init = get_model(cfg.dataset, cfg.device)
            history = run_method(
                method, global_model_init, train_loaders, test_loader,
                devices_scaled, clusters, head_ids, cfg,
            )
            robustness["bandwidth"][bw_scale][method] = max(h["accuracy"] for h in history)
            del global_model_init
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

    cfg.num_rounds = rounds_backup
    return robustness


# ─── Save ─────────────────────────────────────────────────────────────────────

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

        combined_df = pd.concat(seed_dfs)
        combined_df.to_csv(os.path.join(method_dir, "metrics.csv"), index=False)
        all_dfs[method] = seed_dfs
        agg_dfs[method] = aggregate_seeds(seed_dfs)

    summary_df = compute_summary(all_dfs)
    return all_dfs, agg_dfs, summary_df


# ─── Orchestration ────────────────────────────────────────────────────────────

def run(cfg: Config):
    """Run the full pipeline (main experiment + ablations + plots) given a Config."""
    setup_cuda(cfg)

    mode_dir      = os.path.join(cfg.results_dir, cfg.mode)
    summaries_dir = os.path.join(cfg.results_dir, "summaries")
    plots_dir     = os.path.join(mode_dir, "plots")
    os.makedirs(mode_dir,      exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(plots_dir,     exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  HFL-UAV Federated Learning  |  Mode: {cfg.mode.upper()}")
    print(f"  Dataset : {cfg.dataset}  |  Devices : {cfg.num_devices}")
    print(f"  Clusters: {cfg.num_clusters}  |  Rounds  : {cfg.num_rounds}")
    print(f"  Seeds   : {cfg.seeds}  |  IID: {cfg.iid}  |  α: {cfg.alpha}")
    print(f"  Device  : {cfg.device}")
    print(f"{'#'*60}\n")

    t0 = time.time()
    results = run_experiment(cfg)
    print(f"\n[INFO] Main experiment done in {(time.time()-t0)/60:.1f} min.")

    all_dfs, agg_dfs, summary_df = save_results(results, cfg, mode_dir)
    summary_df.to_csv(os.path.join(summaries_dir, f"{cfg.mode}_summary.csv"), index=False)
    print_summary_table(summary_df)

    cluster_lat_dfs = {
        m: get_cluster_latency_stats(seed_hists)
        for m, seed_hists in results.items()
    }

    print("\n[INFO] Running quorum sensitivity sweep…")
    quorum_results = run_quorum_sensitivity(cfg)
    print("\n[INFO] Running scaling analysis…")
    scaling_results = run_scaling(cfg)
    print("\n[INFO] Running robustness sweep…")
    robustness_results = run_robustness(cfg)

    print("\n[INFO] Generating plots…")
    n_plots = generate_all_plots(
        agg_dfs, summary_df, cluster_lat_dfs, plots_dir,
        quorum_results=quorum_results,
        scaling_results=scaling_results,
        robustness_results=robustness_results,
    )

    summary_df.to_csv(os.path.join(summaries_dir, "aggregate.csv"), index=False)

    print(f"\n{'='*60}")
    print("  OUTPUT FILE LOCATIONS")
    print(f"{'='*60}")
    print(f"  Mode results dir : {os.path.abspath(mode_dir)}")
    print(f"  Summary CSV      : {os.path.abspath(os.path.join(summaries_dir, cfg.mode + '_summary.csv'))}")
    print(f"  Plots dir        : {os.path.abspath(plots_dir)}  ({n_plots} plots)")
    print(f"{'='*60}\n")

    return results, summary_df


def run_from_notebook(mode: str = "toy", iid: bool = False, alpha: float = 0.5,
                      device: str = None, results_dir: str = "results"):
    """Convenience wrapper for Jupyter / Kaggle."""
    cfg = Config()
    cfg.mode = mode
    cfg.iid  = iid
    cfg.alpha = alpha
    cfg.results_dir = results_dir
    if device is not None:
        cfg.device = device
    cfg.apply_mode()
    return run(cfg)


def main():
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
