"""
Main entry point for the HFL-UAV study — GPU-native, with toggleable ablations.

CLI:
    python main.py --mode full                # default: main + 3 ablations (no bandwidth)
    python main.py --mode full --only-main    # main experiment only
    python main.py --mode full --no-alpha     # skip alpha sweep
    python main.py --mode full --bandwidth    # include bandwidth sweep too

Notebook:
    from main import run_from_notebook
    run_from_notebook(mode="full")                          # default
    run_from_notebook(mode="full", only_main=True)          # fastest
    run_from_notebook(mode="full", run_bandwidth=True)      # everything
"""

import os
import copy
import time

import numpy as np
import pandas as pd
import torch

from config import parse_args, Config
from data_loader import load_data
from model import get_model
from devices import create_devices
from clustering import build_clustering
from federated import run_method
from metrics import (
    history_to_df, aggregate_seeds, compute_summary,
    print_summary_table, get_cluster_latency_stats,
)
from plotting import generate_all_plots


MAIN_METHODS = [
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
        print("[INFO] Using CPU.")


def _free():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    set_seed(seed)
    global_model_init = get_model(cfg.dataset, cfg.device)

    history = run_method(
        method, global_model_init, train_loaders, test_loader,
        devices, clusters, head_ids, cfg,
    )
    del train_loaders, test_loader, global_model_init
    _free()
    return history


def run_experiment(cfg: Config) -> dict:
    """Main comparison: all 6 methods × all seeds."""
    results = {m: [] for m in MAIN_METHODS}
    for seed in cfg.seeds:
        for method in MAIN_METHODS:
            results[method].append(run_single(method, seed, cfg))
    return results


# ─── Ablation: quorum sensitivity ────────────────────────────────────────────

def run_quorum_sensitivity(cfg: Config) -> dict:
    fractions = [0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
    out = {}

    seed = cfg.seeds[0]
    train_loaders, test_loader = load_data(
        cfg.dataset, cfg.num_devices, cfg.iid, cfg.alpha,
        seed, cfg.batch_size, cfg.test_batch_size, device=cfg.device,
    )
    devices = create_devices(cfg.num_devices, seed)
    head_ids, clusters = build_clustering(devices, cfg.num_clusters, cfg)

    for frac in fractions:
        print(f"\n[Quorum sensitivity] fraction={frac:.2f}")
        out[frac] = {}
        for method in ("topk_quorum", "qsgd_quorum"):
            c2 = copy.copy(cfg); c2.quorum_fraction = frac
            set_seed(seed)
            m_init = get_model(cfg.dataset, cfg.device)
            hist = run_method(method, m_init, train_loaders, test_loader,
                              devices, clusters, head_ids, c2)
            out[frac][method] = (
                max(h["accuracy"] for h in hist),
                float(np.mean([h["latency_round"] for h in hist])),
                float(sum(h["comm_total_mb"] for h in hist)),
            )
            del m_init; _free()
    return out


# ─── Ablation: scaling (num_devices) ─────────────────────────────────────────

def run_scaling(cfg: Config) -> dict:
    device_counts = [10, 20, 30, 40] if cfg.mode == "toy" else [20, 40, 60, 80]
    out = {}
    seed = cfg.seeds[0]
    rounds_backup = cfg.num_rounds
    cfg.num_rounds = max(5, cfg.num_rounds // 4)

    for n in device_counts:
        print(f"\n[Scaling] num_devices={n}")
        c2 = copy.copy(cfg)
        c2.num_devices  = n
        c2.num_clusters = max(2, n // 5)
        out[n] = {}

        train_loaders, test_loader = load_data(
            c2.dataset, n, c2.iid, c2.alpha,
            seed, c2.batch_size, c2.test_batch_size, device=cfg.device,
        )
        devices = create_devices(n, seed)
        head_ids, clusters = build_clustering(devices, c2.num_clusters, c2)

        for method in cfg.ablation_methods:
            set_seed(seed)
            m_init = get_model(c2.dataset, cfg.device)
            hist = run_method(method, m_init, train_loaders, test_loader,
                              devices, clusters, head_ids, c2)
            out[n][method] = max(h["accuracy"] for h in hist)
            del m_init; _free()

    cfg.num_rounds = rounds_backup
    return out


# ─── Ablation: alpha sweep ───────────────────────────────────────────────────

def run_alpha_sweep(cfg: Config) -> dict:
    out = {}
    seed = cfg.seeds[0]
    rounds_backup = cfg.num_rounds
    cfg.num_rounds = max(5, cfg.num_rounds // 4)

    for alpha in [0.1, 0.3, 0.5, 1.0, 5.0]:
        print(f"\n[Robustness/α] alpha={alpha}")
        c2 = copy.copy(cfg); c2.alpha = alpha
        train_loaders, test_loader = load_data(
            c2.dataset, c2.num_devices, False, alpha,
            seed, c2.batch_size, c2.test_batch_size, device=cfg.device,
        )
        devices = create_devices(c2.num_devices, seed)
        head_ids, clusters = build_clustering(devices, c2.num_clusters, c2)
        out[alpha] = {}
        for method in cfg.ablation_methods:
            set_seed(seed)
            m_init = get_model(c2.dataset, cfg.device)
            hist = run_method(method, m_init, train_loaders, test_loader,
                              devices, clusters, head_ids, c2)
            out[alpha][method] = max(h["accuracy"] for h in hist)
            del m_init; _free()

    cfg.num_rounds = rounds_backup
    return out


# ─── Ablation: bandwidth sweep ───────────────────────────────────────────────

def run_bandwidth_sweep(cfg: Config) -> dict:
    out = {}
    seed = cfg.seeds[0]
    rounds_backup = cfg.num_rounds
    cfg.num_rounds = max(5, cfg.num_rounds // 4)

    for bw_scale in [0.5, 0.75, 1.0, 1.5, 2.0]:
        print(f"\n[Robustness/bw] bw_scale={bw_scale}")
        devices_scaled = create_devices(cfg.num_devices, seed)
        for d in devices_scaled:
            d.bandwidth *= bw_scale
        head_ids, clusters = build_clustering(devices_scaled, cfg.num_clusters, cfg)
        train_loaders, test_loader = load_data(
            cfg.dataset, cfg.num_devices, cfg.iid, cfg.alpha,
            seed, cfg.batch_size, cfg.test_batch_size, device=cfg.device,
        )
        out[bw_scale] = {}
        for method in cfg.ablation_methods:
            set_seed(seed)
            m_init = get_model(cfg.dataset, cfg.device)
            hist = run_method(method, m_init, train_loaders, test_loader,
                              devices_scaled, clusters, head_ids, cfg)
            out[bw_scale][method] = max(h["accuracy"] for h in hist)
            del m_init; _free()

    cfg.num_rounds = rounds_backup
    return out


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

        pd.concat(seed_dfs).to_csv(os.path.join(method_dir, "metrics.csv"), index=False)
        all_dfs[method] = seed_dfs
        agg_dfs[method] = aggregate_seeds(seed_dfs)

    summary_df = compute_summary(all_dfs)
    return all_dfs, agg_dfs, summary_df


# ─── Plan summary printer ────────────────────────────────────────────────────

def _print_plan(cfg: Config):
    n_main = len(cfg.seeds) * len(MAIN_METHODS)
    n_q  = 6 * 2                       if cfg.run_quorum    else 0
    n_s  = 4 * len(cfg.ablation_methods) if cfg.run_scaling   else 0
    n_a  = 5 * len(cfg.ablation_methods) if cfg.run_alpha     else 0
    n_bw = 5 * len(cfg.ablation_methods) if cfg.run_bandwidth else 0
    total = n_main + n_q + n_s + n_a + n_bw

    print(f"\n{'─'*60}")
    print("  RUN PLAN")
    print(f"{'─'*60}")
    print(f"  Main experiment    : {n_main:3d} trainings  ({len(cfg.seeds)} seeds × {len(MAIN_METHODS)} methods)")
    print(f"  Quorum sensitivity : {n_q:3d} trainings  [{'on'  if cfg.run_quorum    else 'off'}]")
    print(f"  Scaling            : {n_s:3d} trainings  [{'on'  if cfg.run_scaling   else 'off'}]")
    print(f"  Alpha sweep        : {n_a:3d} trainings  [{'on'  if cfg.run_alpha     else 'off'}]")
    print(f"  Bandwidth sweep    : {n_bw:3d} trainings  [{'on'  if cfg.run_bandwidth else 'off'}]")
    print(f"  Ablation methods   : {cfg.ablation_methods}")
    print(f"  TOTAL              : {total:3d} trainings")
    print(f"{'─'*60}\n")


# ─── Orchestration ────────────────────────────────────────────────────────────

def run(cfg: Config):
    setup_cuda(cfg)

    mode_dir      = os.path.join(cfg.results_dir, cfg.mode)
    summaries_dir = os.path.join(cfg.results_dir, "summaries")
    plots_dir     = os.path.join(mode_dir, "plots")
    for d in (mode_dir, summaries_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  HFL-UAV Federated Learning  |  Mode: {cfg.mode.upper()}")
    print(f"  Dataset : {cfg.dataset}  |  Devices: {cfg.num_devices}")
    print(f"  Clusters: {cfg.num_clusters}  |  Rounds : {cfg.num_rounds}")
    print(f"  Seeds   : {cfg.seeds}  |  IID: {cfg.iid}  |  α: {cfg.alpha}")
    print(f"  Device  : {cfg.device}")
    print(f"{'#'*60}")
    _print_plan(cfg)

    # ── Main experiment ──────────────────────────────────────────────────
    t0 = time.time()
    results = run_experiment(cfg)
    t_main = time.time() - t0
    print(f"\n[INFO] Main experiment: {t_main/60:.1f} min.")

    all_dfs, agg_dfs, summary_df = save_results(results, cfg, mode_dir)
    summary_df.to_csv(os.path.join(summaries_dir, f"{cfg.mode}_summary.csv"), index=False)
    print_summary_table(summary_df)

    cluster_lat_dfs = {
        m: get_cluster_latency_stats(seed_hists)
        for m, seed_hists in results.items()
    }

    # ── Ablations (each gated) ───────────────────────────────────────────
    quorum_results = scaling_results = alpha_results = bandwidth_results = None

    if cfg.run_quorum:
        print("\n[INFO] Running quorum sensitivity sweep…")
        t = time.time()
        quorum_results = run_quorum_sensitivity(cfg)
        print(f"[INFO] Quorum sweep: {(time.time()-t)/60:.1f} min.")

    if cfg.run_scaling:
        print("\n[INFO] Running scaling analysis…")
        t = time.time()
        scaling_results = run_scaling(cfg)
        print(f"[INFO] Scaling sweep: {(time.time()-t)/60:.1f} min.")

    if cfg.run_alpha:
        print("\n[INFO] Running α robustness sweep…")
        t = time.time()
        alpha_results = run_alpha_sweep(cfg)
        print(f"[INFO] α sweep: {(time.time()-t)/60:.1f} min.")

    if cfg.run_bandwidth:
        print("\n[INFO] Running bandwidth robustness sweep…")
        t = time.time()
        bandwidth_results = run_bandwidth_sweep(cfg)
        print(f"[INFO] Bandwidth sweep: {(time.time()-t)/60:.1f} min.")

    # Combine α + bw under the same key plotting.py expects
    robustness_results = None
    if alpha_results is not None or bandwidth_results is not None:
        robustness_results = {"alpha": alpha_results or {},
                              "bandwidth": bandwidth_results or {}}

    # ── Plots ────────────────────────────────────────────────────────────
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


# ─── Notebook wrapper ────────────────────────────────────────────────────────

def run_from_notebook(
    mode: str = "toy",
    iid: bool = False,
    alpha: float = 0.5,
    device: str = None,
    results_dir: str = "results",
    seeds: list = None,
    *,
    only_main: bool = False,
    run_quorum: bool = True,
    run_scaling: bool = True,
    run_alpha: bool = True,
    run_bandwidth: bool = False,
    ablation_methods: list = None,
):
    """
    Convenience wrapper for Kaggle / Jupyter.

    Examples:
        run_from_notebook(mode="full")                                  # default
        run_from_notebook(mode="full", only_main=True)                  # fastest
        run_from_notebook(mode="full", run_alpha=False)                 # skip α
        run_from_notebook(mode="full", run_bandwidth=True)              # everything
        run_from_notebook(mode="full", seeds=[42, 7, 123])              # 3 seeds not 5
    """
    cfg = Config()
    cfg.mode  = mode
    cfg.iid   = iid
    cfg.alpha = alpha
    cfg.results_dir = results_dir
    if device is not None:
        cfg.device = device

    cfg.apply_mode()
    if seeds is not None:
        cfg.seeds = seeds

    cfg.run_quorum    = run_quorum
    cfg.run_scaling   = run_scaling
    cfg.run_alpha     = run_alpha
    cfg.run_bandwidth = run_bandwidth
    if only_main:
        cfg.run_quorum = cfg.run_scaling = cfg.run_alpha = cfg.run_bandwidth = False

    if ablation_methods is not None:
        cfg.ablation_methods = ablation_methods

    return run(cfg)


def main():
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
