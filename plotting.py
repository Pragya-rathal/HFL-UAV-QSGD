"""
IEEE Transactions-level plotting module.
Generates all required plots from experiment results.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# ─── Style ────────────────────────────────────────────────────────────────────

COLORS = {
    "standard_fl":  "#1f77b4",
    "clustered_fl": "#ff7f0e",
    "topk_ef":      "#2ca02c",
    "qsgd":         "#d62728",
    "topk_quorum":  "#9467bd",
    "qsgd_quorum":  "#8c564b",
}
MARKERS = {
    "standard_fl":  "o",
    "clustered_fl": "s",
    "topk_ef":      "^",
    "qsgd":         "D",
    "topk_quorum":  "P",
    "qsgd_quorum":  "*",
}
LABELS = {
    "standard_fl":  "A: Std-FL",
    "clustered_fl": "B: Clustered-FL",
    "topk_ef":      "C: Top-K+EF",
    "qsgd":         "D: QSGD",
    "topk_quorum":  "E: Top-K+Quorum",
    "qsgd_quorum":  "F: QSGD+Quorum",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "grid.alpha": 0.4,
})

METHODS = list(COLORS.keys())


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _method_mean_std(agg_dfs: Dict[str, pd.DataFrame], col: str):
    """Return {method: (rounds, means, stds)}."""
    out = {}
    for m, df in agg_dfs.items():
        rounds = df["round"].values
        mean_col = col + "_mean"
        std_col = col + "_std"
        means = df[mean_col].values if mean_col in df.columns else df[col].values
        stds = df[std_col].values if std_col in df.columns else np.zeros_like(means)
        out[m] = (rounds, means, stds)
    return out


def _plot_with_band(ax, rounds, means, stds, method):
    c = COLORS[method]
    mk = MARKERS[method]
    lb = LABELS[method]
    step = max(1, len(rounds) // 10)
    ax.plot(rounds, means, color=c, marker=mk, markevery=step, label=lb)
    ax.fill_between(rounds, means - stds, means + stds, alpha=0.15, color=c)


# ─── Core Plots ──────────────────────────────────────────────────────────────

def plot_accuracy_vs_rounds(agg_dfs, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = _method_mean_std(agg_dfs, "accuracy")
    for m in METHODS:
        if m in data:
            _plot_with_band(ax, *data[m], m)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy vs. Communication Rounds")
    ax.legend(loc="lower right")
    ax.grid(True)
    _save(fig, os.path.join(plots_dir, "accuracy_vs_rounds.pdf"))


def plot_loss_vs_rounds(agg_dfs, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = _method_mean_std(agg_dfs, "eval_loss")
    for m in METHODS:
        if m in data:
            _plot_with_band(ax, *data[m], m)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Loss")
    ax.set_title("Test Loss vs. Communication Rounds")
    ax.legend(loc="upper right")
    ax.grid(True)
    _save(fig, os.path.join(plots_dir, "loss_vs_rounds.pdf"))


def plot_latency_vs_rounds(agg_dfs, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = _method_mean_std(agg_dfs, "latency_round")
    for m in METHODS:
        if m in data:
            _plot_with_band(ax, *data[m], m)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Round Latency (s)")
    ax.set_title("Round Latency vs. Communication Rounds")
    ax.legend(loc="upper right")
    ax.grid(True)
    _save(fig, os.path.join(plots_dir, "latency_vs_rounds.pdf"))


def plot_comm_vs_rounds(agg_dfs, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    # Cumulative comm
    for m, df in agg_dfs.items():
        col = "comm_total_mb_mean" if "comm_total_mb_mean" in df.columns else "comm_total_mb"
        cum = df[col].cumsum().values
        rounds = df["round"].values
        ax.plot(rounds, cum, color=COLORS[m], marker=MARKERS[m],
                markevery=max(1, len(rounds) // 10), label=LABELS[m])
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Cumulative Communication (MB)")
    ax.set_title("Cumulative Communication vs. Rounds")
    ax.legend(loc="upper left")
    ax.grid(True)
    _save(fig, os.path.join(plots_dir, "comm_vs_rounds.pdf"))


# ─── Tradeoff Plots ───────────────────────────────────────────────────────────

def plot_accuracy_vs_latency(summary_df, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in summary_df.iterrows():
        m = row["method"]
        ax.errorbar(
            row["avg_latency_mean"], row["best_acc_mean"],
            xerr=row["avg_latency_std"], yerr=row["best_acc_std"],
            fmt=MARKERS[m], color=COLORS[m], label=LABELS[m],
            markersize=8, capsize=4,
        )
    ax.set_xlabel("Average Round Latency (s)")
    ax.set_ylabel("Best Test Accuracy")
    ax.set_title("Accuracy–Latency Tradeoff")
    ax.legend(loc="lower right")
    ax.grid(True)
    _save(fig, os.path.join(plots_dir, "accuracy_vs_latency.pdf"))


def plot_accuracy_vs_comm(summary_df, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in summary_df.iterrows():
        m = row["method"]
        ax.errorbar(
            row["total_comm_mb_mean"], row["best_acc_mean"],
            xerr=row["total_comm_mb_std"], yerr=row["best_acc_std"],
            fmt=MARKERS[m], color=COLORS[m], label=LABELS[m],
            markersize=8, capsize=4,
        )
    ax.set_xlabel("Total Communication (MB)")
    ax.set_ylabel("Best Test Accuracy")
    ax.set_title("Accuracy–Communication Tradeoff")
    ax.legend(loc="lower right")
    ax.grid(True)
    _save(fig, os.path.join(plots_dir, "accuracy_vs_comm.pdf"))


def plot_comm_vs_latency(summary_df, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in summary_df.iterrows():
        m = row["method"]
        ax.errorbar(
            row["avg_latency_mean"], row["total_comm_mb_mean"],
            xerr=row["avg_latency_std"], yerr=row["total_comm_mb_std"],
            fmt=MARKERS[m], color=COLORS[m], label=LABELS[m],
            markersize=8, capsize=4,
        )
    ax.set_xlabel("Average Round Latency (s)")
    ax.set_ylabel("Total Communication (MB)")
    ax.set_title("Communication–Latency Tradeoff")
    ax.legend(loc="upper left")
    ax.grid(True)
    _save(fig, os.path.join(plots_dir, "comm_vs_latency.pdf"))


# ─── System Behaviour ─────────────────────────────────────────────────────────

def plot_active_devices(agg_dfs, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, df in agg_dfs.items():
        col = "active_devices_mean" if "active_devices_mean" in df.columns else "active_devices"
        ax.plot(df["round"].values, df[col].values,
                color=COLORS[m], marker=MARKERS[m],
                markevery=max(1, len(df) // 10), label=LABELS[m])
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Active Devices")
    ax.set_title("Active Devices per Round")
    ax.legend(loc="upper right")
    ax.grid(True)
    _save(fig, os.path.join(plots_dir, "active_devices.pdf"))


def plot_latency_distribution(agg_dfs, plots_dir):
    """Box plot of per-round latencies across all rounds."""
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(METHODS))
    for i, m in enumerate(METHODS):
        if m not in agg_dfs:
            continue
        df = agg_dfs[m]
        col = "latency_round_mean" if "latency_round_mean" in df.columns else "latency_round"
        vals = df[col].values
        bp = ax.boxplot(vals, positions=[i], widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=COLORS[m], alpha=0.6))
    ax.set_xticks(positions[:len([m for m in METHODS if m in agg_dfs])])
    ax.set_xticklabels(
        [LABELS[m] for m in METHODS if m in agg_dfs],
        rotation=20, ha="right"
    )
    ax.set_ylabel("Round Latency (s)")
    ax.set_title("Latency Distribution Across Rounds")
    ax.grid(True, axis="y")
    _save(fig, os.path.join(plots_dir, "latency_distribution.pdf"))


def plot_cluster_latency(cluster_lat_dfs: Dict[str, pd.DataFrame], plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric, ylabel) in zip(
        axes,
        [("cluster_latency", "Cluster Latency (s)")]
    ):
        for m, df in cluster_lat_dfs.items():
            if df.empty:
                continue
            per_round = df.groupby("round")["cluster_latency"].mean()
            ax.plot(per_round.index, per_round.values,
                    color=COLORS[m], label=LABELS[m])
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Mean Cluster Latency (s)")
    axes[0].set_title("Mean Cluster-Level Latency per Round")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)
    # Second panel: distribution
    data = []
    labels = []
    for m, df in cluster_lat_dfs.items():
        if not df.empty:
            data.append(df["cluster_latency"].values)
            labels.append(LABELS[m])
    if data:
        axes[1].boxplot(data, labels=labels)
        axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Cluster Latency (s)")
    axes[1].set_title("Cluster Latency Distribution")
    axes[1].grid(True)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "cluster_latency.pdf"))


# ─── Advanced Plots ───────────────────────────────────────────────────────────

def plot_efficiency_metrics(summary_df, plots_dir):
    """Accuracy-per-MB and accuracy-per-second efficiency."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    methods = summary_df["method"].tolist()
    acc_per_mb = summary_df["best_acc_mean"] / (summary_df["total_comm_mb_mean"] + 1e-6)
    acc_per_sec = summary_df["best_acc_mean"] / (summary_df["avg_latency_mean"] + 1e-6)

    for ax, vals, title, ylabel in [
        (axes[0], acc_per_mb.values, "Communication Efficiency", "Accuracy / MB"),
        (axes[1], acc_per_sec.values, "Latency Efficiency", "Accuracy / s"),
    ]:
        colors = [COLORS[m] for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=22, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")

    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "efficiency_metrics.pdf"))


def plot_compression_analysis(agg_dfs, plots_dir):
    """Compare compression ratio: comm_total_mb relative to standard_fl."""
    if "standard_fl" not in agg_dfs:
        return
    baseline = agg_dfs["standard_fl"]["comm_total_mb_mean" if "comm_total_mb_mean" in agg_dfs["standard_fl"].columns else "comm_total_mb"].sum()

    ratios = {}
    for m, df in agg_dfs.items():
        col = "comm_total_mb_mean" if "comm_total_mb_mean" in df.columns else "comm_total_mb"
        ratios[m] = df[col].sum() / (baseline + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 4))
    ms = list(ratios.keys())
    vals = [ratios[m] for m in ms]
    colors = [COLORS[m] for m in ms]
    ax.bar(range(len(ms)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Std-FL baseline")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([LABELS[m] for m in ms], rotation=22, ha="right")
    ax.set_ylabel("Communication Ratio (vs Std-FL)")
    ax.set_title("Compression Analysis – Relative Communication Cost")
    ax.legend()
    ax.grid(True, axis="y")
    _save(fig, os.path.join(plots_dir, "compression_analysis.pdf"))


def plot_quorum_sensitivity(quorum_results: Dict[float, Dict], plots_dir):
    """
    quorum_results: {fraction → {method → (acc, lat, comm)}}
    Shows how accuracy and latency change with quorum fraction.
    """
    if not quorum_results:
        return
    fracs = sorted(quorum_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in ("topk_quorum", "qsgd_quorum"):
        accs = [quorum_results[f].get(m, (0, 0, 0))[0] for f in fracs]
        lats = [quorum_results[f].get(m, (0, 0, 0))[1] for f in fracs]
        axes[0].plot(fracs, accs, marker="o", color=COLORS[m], label=LABELS[m])
        axes[1].plot(fracs, lats, marker="s", color=COLORS[m], label=LABELS[m])
    axes[0].set_xlabel("Quorum Fraction")
    axes[0].set_ylabel("Best Accuracy")
    axes[0].set_title("Quorum Sensitivity – Accuracy")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].set_xlabel("Quorum Fraction")
    axes[1].set_ylabel("Avg Round Latency (s)")
    axes[1].set_title("Quorum Sensitivity – Latency")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "quorum_sensitivity.pdf"))


def plot_scaling_analysis(scaling_results: Dict[int, Dict], plots_dir):
    """
    scaling_results: {num_devices → {method → best_acc}}
    """
    if not scaling_results:
        return
    ns = sorted(scaling_results.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in METHODS:
        accs = [scaling_results[n].get(m, np.nan) for n in ns]
        ax.plot(ns, accs, marker=MARKERS[m], color=COLORS[m], label=LABELS[m])
    ax.set_xlabel("Number of Devices")
    ax.set_ylabel("Best Accuracy")
    ax.set_title("Scaling Analysis")
    ax.legend(fontsize=7)
    ax.grid(True)
    _save(fig, os.path.join(plots_dir, "scaling_analysis.pdf"))


def plot_robustness(robustness_results: Dict, plots_dir):
    """
    robustness_results: {"alpha": {alpha → {method → acc}},
                         "bandwidth": {bw_scale → {method → acc}}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key, xlabel in [
        (axes[0], "alpha", "Dirichlet α (data heterogeneity)"),
        (axes[1], "bandwidth", "Bandwidth Scale Factor"),
    ]:
        if key not in robustness_results:
            continue
        sweep = robustness_results[key]
        xs = sorted(sweep.keys())
        for m in METHODS:
            ys = [sweep[x].get(m, np.nan) for x in xs]
            ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m], label=LABELS[m])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Best Accuracy")
        ax.set_title(f"Robustness to {xlabel}")
        ax.legend(fontsize=7)
        ax.grid(True)

    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "robustness.pdf"))


def plot_convergence_metrics(summary_df, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    methods = summary_df["method"].tolist()

    # Convergence round
    axes[0].bar(
        range(len(methods)),
        summary_df["convergence_round_mean"].values,
        yerr=summary_df["convergence_round_std"].values,
        color=[COLORS[m] for m in methods],
        edgecolor="black", linewidth=0.5, capsize=4,
    )
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels([LABELS[m] for m in methods], rotation=22, ha="right", fontsize=8)
    axes[0].set_ylabel("Rounds to 95% Best Accuracy")
    axes[0].set_title("Convergence Speed")
    axes[0].grid(True, axis="y")

    # Best accuracy
    axes[1].bar(
        range(len(methods)),
        summary_df["best_acc_mean"].values,
        yerr=summary_df["best_acc_std"].values,
        color=[COLORS[m] for m in methods],
        edgecolor="black", linewidth=0.5, capsize=4,
    )
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels([LABELS[m] for m in methods], rotation=22, ha="right", fontsize=8)
    axes[1].set_ylabel("Best Test Accuracy")
    axes[1].set_title("Final Accuracy Comparison")
    axes[1].grid(True, axis="y")

    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "convergence_metrics.pdf"))


def plot_3d_tradeoff(summary_df, plots_dir):
    """3-D scatter: latency × communication × accuracy."""
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    for _, row in summary_df.iterrows():
        m = row["method"]
        ax.scatter(
            row["avg_latency_mean"],
            row["total_comm_mb_mean"],
            row["best_acc_mean"],
            c=COLORS[m], marker=MARKERS[m], s=80, label=LABELS[m],
        )

    ax.set_xlabel("Avg Latency (s)", labelpad=8)
    ax.set_ylabel("Total Comm (MB)", labelpad=8)
    ax.set_zlabel("Best Accuracy", labelpad=8)
    ax.set_title("3D Tradeoff: Latency × Comm × Accuracy")
    ax.legend(loc="upper left", fontsize=7)
    _save(fig, os.path.join(plots_dir, "3d_tradeoff.pdf"))


def plot_radar(summary_df, plots_dir):
    """Radar / spider chart comparing methods across 4 normalised axes."""
    categories = ["Accuracy", "1/Latency", "1/Comm", "Conv Speed"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Normalise each axis to [0,1]
    acc_norm = (summary_df["best_acc_mean"] - summary_df["best_acc_mean"].min()) / \
               (summary_df["best_acc_mean"].max() - summary_df["best_acc_mean"].min() + 1e-8)
    lat_inv = 1 / (summary_df["avg_latency_mean"] + 1e-8)
    lat_norm = (lat_inv - lat_inv.min()) / (lat_inv.max() - lat_inv.min() + 1e-8)
    comm_inv = 1 / (summary_df["total_comm_mb_mean"] + 1e-8)
    comm_norm = (comm_inv - comm_inv.min()) / (comm_inv.max() - comm_inv.min() + 1e-8)
    conv_inv = 1 / (summary_df["convergence_round_mean"] + 1e-8)
    conv_norm = (conv_inv - conv_inv.min()) / (conv_inv.max() - conv_inv.min() + 1e-8)

    for i, row in summary_df.iterrows():
        m = row["method"]
        vals = [acc_norm.iloc[i], lat_norm.iloc[i], comm_norm.iloc[i], conv_norm.iloc[i]]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[m], linewidth=1.5, label=LABELS[m])
        ax.fill(angles, vals, color=COLORS[m], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Method Radar Chart\n(normalised, higher = better)", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    _save(fig, os.path.join(plots_dir, "radar_chart.pdf"))


# ─── Master plotting function ─────────────────────────────────────────────────

def generate_all_plots(
    agg_dfs: Dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    cluster_lat_dfs: Dict[str, pd.DataFrame],
    plots_dir: str,
    quorum_results: Optional[Dict] = None,
    scaling_results: Optional[Dict] = None,
    robustness_results: Optional[Dict] = None,
) -> int:
    os.makedirs(plots_dir, exist_ok=True)
    count = 0

    def _run(fn, *args):
        nonlocal count
        try:
            fn(*args)
            count += 1
        except Exception as e:
            print(f"  [plot warning] {fn.__name__}: {e}")

    _run(plot_accuracy_vs_rounds, agg_dfs, plots_dir)
    _run(plot_loss_vs_rounds, agg_dfs, plots_dir)
    _run(plot_latency_vs_rounds, agg_dfs, plots_dir)
    _run(plot_comm_vs_rounds, agg_dfs, plots_dir)
    _run(plot_accuracy_vs_latency, summary_df, plots_dir)
    _run(plot_accuracy_vs_comm, summary_df, plots_dir)
    _run(plot_comm_vs_latency, summary_df, plots_dir)
    _run(plot_active_devices, agg_dfs, plots_dir)
    _run(plot_latency_distribution, agg_dfs, plots_dir)
    _run(plot_cluster_latency, cluster_lat_dfs, plots_dir)
    _run(plot_efficiency_metrics, summary_df, plots_dir)
    _run(plot_compression_analysis, agg_dfs, plots_dir)
    _run(plot_convergence_metrics, summary_df, plots_dir)
    _run(plot_3d_tradeoff, summary_df, plots_dir)
    _run(plot_radar, summary_df, plots_dir)
    if quorum_results:
        _run(plot_quorum_sensitivity, quorum_results, plots_dir)
    if scaling_results:
        _run(plot_scaling_analysis, scaling_results, plots_dir)
    if robustness_results:
        _run(plot_robustness, robustness_results, plots_dir)

    return count
