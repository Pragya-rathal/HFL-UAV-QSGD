"""
plotting_topoco.py — Publication-grade plots specific to the topology-
adaptive primal-dual framework.

ADDITIVE: this module does not touch plotting.py. The existing per-round
plots (accuracy, latency, communication, etc.) keep working for every
method including topoco and fedprox, because they all emit the same base
schema.  This file adds plots that only make sense for topoco's extended
schema (λ_*, q̄, density, etc.) or that compare adaptivity variants.

All figures use a consistent style suitable for IEEE Transactions figures:
single-column width, 300 dpi, grids at 30 % alpha, no excessive colour.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _style():
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "axes.grid": True, "grid.alpha": 0.3,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dual multiplier evolution — the visual heart of the contribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_lambda_trajectories(topoco_df: pd.DataFrame, save_path: str):
    _style()
    if "lambda_C" not in topoco_df.columns:
        return
    g = topoco_df.groupby("round").agg(
        {"lambda_C": ["mean", "std"], "lambda_L": ["mean", "std"], "lambda_D": ["mean", "std"]}
    ).reset_index()
    g.columns = ["round",
                 "lC_mean", "lC_std", "lL_mean", "lL_std", "lD_mean", "lD_std"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for tag, mean, std, marker, color in [
        ("$\\lambda_C$ (bandwidth)", "lC_mean", "lC_std", "o", "C0"),
        ("$\\lambda_L$ (latency)",   "lL_mean", "lL_std", "s", "C1"),
        ("$\\lambda_D$ (divergence)", "lD_mean", "lD_std", "^", "C3"),
    ]:
        ax.plot(g["round"], g[mean], marker=marker, label=tag, color=color)
        ax.fill_between(g["round"], g[mean] - g[std], g[mean] + g[std],
                        color=color, alpha=0.15)
    ax.set_xlabel("Communication round t")
    ax.set_ylabel("Dual variable")
    ax.set_title("Lagrangian multiplier trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Adaptive compression trajectory
# ─────────────────────────────────────────────────────────────────────────────

def plot_compression_trajectory(topoco_df: pd.DataFrame, save_path: str):
    _style()
    if "sched_q_mean" not in topoco_df.columns:
        return
    g = topoco_df.groupby("round").agg({"sched_q_mean": ["mean", "std"]}).reset_index()
    g.columns = ["round", "q_mean", "q_std"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(g["round"], g["q_mean"], "-o", color="C2", label=r"$\bar q_t$")
    ax.fill_between(g["round"], g["q_mean"] - g["q_std"], g["q_mean"] + g["q_std"],
                    color="C2", alpha=0.15)
    ax.set_xlabel("Communication round t")
    ax.set_ylabel(r"Mean kept fraction $\bar q_t$")
    ax.set_title("Adaptive compression")
    ax.set_ylim(0, max(0.6, float(g["q_mean"].max() * 1.2)))
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Topology evolution (density + active-device count + CC)
# ─────────────────────────────────────────────────────────────────────────────

def plot_topology_evolution(topoco_df: pd.DataFrame, save_path: str):
    _style()
    if "density" not in topoco_df.columns:
        return
    g = topoco_df.groupby("round").agg({
        "density": "mean", "active_devices": "mean", "avg_local_cc": "mean",
    }).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(g["round"], g["density"], "-o", color="C3")
    axes[0].set_title("Graph density"); axes[0].set_xlabel("Round"); axes[0].set_ylabel(r"$\rho(G_t)$")
    axes[1].plot(g["round"], g["active_devices"], "-s", color="C4")
    axes[1].set_title("Active devices"); axes[1].set_xlabel("Round"); axes[1].set_ylabel(r"$|V_t|$")
    axes[2].plot(g["round"], g["avg_local_cc"], "-^", color="C5")
    axes[2].set_title("Average local CC"); axes[2].set_xlabel("Round"); axes[2].set_ylabel("CC")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Communication–accuracy Pareto across methods
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_comm_acc(summary_df: pd.DataFrame, save_path: str):
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, row in summary_df.iterrows():
        label = str(row.get("label", row["method"]))
        is_topoco = "topoco" in row["method"].lower()
        ax.scatter(row["total_comm_mb_mean"], row["best_acc_mean"],
                   s=180 if is_topoco else 90,
                   marker="*" if is_topoco else "o",
                   edgecolors="black", linewidths=1.0, label=label, zorder=3)
        ax.annotate(label.split(":")[0],
                    (row["total_comm_mb_mean"], row["best_acc_mean"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Total communication (MB)")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Communication–accuracy Pareto")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Adaptivity ablation — bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_adaptivity_ablation(ablation: Dict[str, float], save_path: str,
                             metric_name: str = "Best test accuracy"):
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = list(ablation.keys())
    vals  = list(ablation.values())
    bars = ax.bar(names, vals, color=["C0", "C1", "C2", "C3"][:len(names)],
                  edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric_name)
    ax.set_title("Adaptivity ablation (compression × participation)")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stress-sweep: accuracy under stress + adaptivity gap
# ─────────────────────────────────────────────────────────────────────────────

def plot_stress_sweep(stress_results: Dict[float, Dict[str, tuple]], save_path: str):
    """
    Two-panel figure.

    LEFT  — line plot, x = dropout probability, y = best accuracy, one line
             per method.  Tells the reader how each method degrades as the
             network gets less reliable.

    RIGHT — bar plot, x = dropout level, y = Acc(topoco_full) − Acc(topoco_static).
             The gap widening to the right is the central piece of evidence
             that the adaptive primal-dual mechanism (and not just clustering
             or compression) is what's earning the contribution.
    """
    _style()
    if not stress_results:
        return

    dropouts = sorted(stress_results.keys())
    methods  = ["clustered_fl", "hierfavg", "topoco_static", "topoco_full"]
    labels   = {
        "clustered_fl":  "Clustered FL",
        "hierfavg":      "HierFAVG",
        "topoco_static": "TopoCo (static)",
        "topoco_full":   "TopoCo (full, ours)",
    }
    colors  = ["C0", "C1", "C2", "C3"]
    markers = ["o", "s", "^", "*"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    for method, color, marker in zip(methods, colors, markers):
        accs = [stress_results[d].get(method, (np.nan, np.nan, np.nan))[0] for d in dropouts]
        ax1.plot(dropouts, accs, "-" + marker, color=color, label=labels[method],
                 markersize=8 if marker == "*" else 6,
                 linewidth=2.0 if method == "topoco_full" else 1.4)
    ax1.set_xlabel("Transient dropout probability  $p_{\\rm drop}$")
    ax1.set_ylabel("Best test accuracy")
    ax1.set_title("Accuracy under network stress")
    ax1.legend(fontsize=9)

    gap = []
    for d in dropouts:
        full = stress_results[d].get("topoco_full",   (np.nan, 0, 0))[0]
        stat = stress_results[d].get("topoco_static", (np.nan, 0, 0))[0]
        gap.append(full - stat)
    bars = ax2.bar(range(len(dropouts)), gap, color="C3",
                   edgecolor="black", linewidth=0.8)
    ax2.set_xticks(range(len(dropouts)))
    ax2.set_xticklabels([f"{d:.2f}" for d in dropouts])
    ax2.set_xlabel("Dropout probability  $p_{\\rm drop}$")
    ax2.set_ylabel(r"$\Delta$Acc  (full − static)")
    ax2.set_title("Adaptivity gap widens under stress")
    ax2.axhline(0, color="black", linewidth=0.5)
    for b, g in zip(bars, gap):
        ax2.text(b.get_x() + b.get_width() / 2, g + 0.003, f"{g:+.3f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────────────────────

def generate_topoco_plots(per_method_dfs: Dict[str, List[pd.DataFrame]],
                          summary_df: pd.DataFrame,
                          ablation_results: Dict[str, float],
                          plots_dir: str,
                          stress_results: Dict[float, Dict[str, tuple]] = None) -> int:
    """
    `per_method_dfs[method]` is the list of per-seed DataFrames (as built by
    main.save_results). Returns the number of figures emitted.
    """
    os.makedirs(plots_dir, exist_ok=True)
    n = 0
    if "topoco" in per_method_dfs and per_method_dfs["topoco"]:
        topoco_df = pd.concat(per_method_dfs["topoco"])
        plot_lambda_trajectories(topoco_df,    os.path.join(plots_dir, "topoco_lambdas.png"));        n += 1
        plot_compression_trajectory(topoco_df, os.path.join(plots_dir, "topoco_compression.png"));    n += 1
        plot_topology_evolution(topoco_df,     os.path.join(plots_dir, "topoco_topology.png"));        n += 1
    if summary_df is not None and len(summary_df) > 0:
        plot_pareto_comm_acc(summary_df,       os.path.join(plots_dir, "pareto_comm_acc.png"));        n += 1
    if ablation_results:
        plot_adaptivity_ablation(ablation_results, os.path.join(plots_dir, "adaptivity_ablation.png"));n += 1
    if stress_results:
        plot_stress_sweep(stress_results,      os.path.join(plots_dir, "stress_sweep.png"));           n += 1
    return n
