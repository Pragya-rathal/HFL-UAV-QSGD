from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_method_plots(df: pd.DataFrame, out_dir: Path, method: str) -> int:
    plot_count = 0
    r = df["round"]

    fig, ax = plt.subplots()
    ax.plot(r, df["accuracy_mean"])
    ax.set_title(f"Accuracy vs Rounds ({method})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    _save(fig, out_dir / "accuracy_vs_rounds.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.plot(r, df["loss_mean"], color="orange")
    ax.set_title(f"Loss vs Rounds ({method})")
    _save(fig, out_dir / "loss_vs_rounds.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.plot(r, df["latency_mean"], color="green")
    ax.fill_between(r, df["latency_p75"], df["max_latency_mean"], alpha=0.2)
    ax.set_title(f"Latency vs Rounds ({method})")
    _save(fig, out_dir / "latency_vs_rounds.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.plot(r, df["comm_mean"], color="red")
    ax.set_title(f"Communication vs Rounds ({method})")
    _save(fig, out_dir / "communication_vs_rounds.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.scatter(df["latency_mean"], df["accuracy_mean"], s=10)
    ax.set_xlabel("Latency")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Latency")
    _save(fig, out_dir / "accuracy_latency_tradeoff.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.scatter(df["comm_mean"], df["accuracy_mean"], s=10)
    ax.set_xlabel("Communication (MB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Communication")
    _save(fig, out_dir / "accuracy_communication_tradeoff.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.scatter(df["comm_mean"], df["latency_mean"], s=10)
    ax.set_xlabel("Communication (MB)")
    ax.set_ylabel("Latency")
    ax.set_title("Communication vs Latency")
    _save(fig, out_dir / "communication_latency_tradeoff.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.plot(r, df["active_devices_mean"], color="purple")
    ax.set_title("Active Devices per Round")
    _save(fig, out_dir / "active_devices.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.hist(df["latency_mean"], bins=15)
    ax.set_title("Latency Distribution")
    _save(fig, out_dir / "latency_distribution.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.plot(r, df["max_latency_mean"] - df["latency_mean"])
    ax.set_title("Cluster Latency Spread")
    _save(fig, out_dir / "cluster_latency_spread.png")
    plot_count += 1

    fig, ax = plt.subplots()
    efficiency = df["accuracy_mean"] / (df["latency_mean"] + 1e-9)
    ax.plot(r, efficiency)
    ax.set_title("Efficiency: Accuracy/Latency")
    _save(fig, out_dir / "efficiency_metric.png")
    plot_count += 1

    fig, ax = plt.subplots()
    comp_gain = (df["comm_mean"].iloc[0] + 1e-9) / (df["comm_mean"] + 1e-9)
    ax.plot(r, comp_gain)
    ax.set_title("Compression Gain")
    _save(fig, out_dir / "compression_analysis.png")
    plot_count += 1

    fig, ax = plt.subplots()
    quorum_proxy = df["active_devices_mean"] / df["active_devices_mean"].max()
    ax.plot(r, quorum_proxy)
    ax.set_title("Quorum Sensitivity Proxy")
    _save(fig, out_dir / "quorum_sensitivity.png")
    plot_count += 1

    fig, ax = plt.subplots()
    scale_proxy = df["active_devices_mean"] / (df["latency_mean"] + 1e-9)
    ax.plot(r, scale_proxy)
    ax.set_title("Scaling Proxy")
    _save(fig, out_dir / "scaling_analysis.png")
    plot_count += 1

    fig, ax = plt.subplots()
    robust_proxy = df["accuracy_std"] / (df["latency_std"] + 1e-6)
    ax.plot(r, robust_proxy)
    ax.set_title("Robustness Proxy (alpha/bandwidth)")
    _save(fig, out_dir / "robustness_proxy.png")
    plot_count += 1

    fig, ax = plt.subplots()
    ax.plot(r, np.gradient(df["accuracy_mean"].values))
    ax.set_title("Convergence Gradient")
    _save(fig, out_dir / "convergence_metric.png")
    plot_count += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(df["comm_mean"], df["latency_mean"], df["accuracy_mean"])
    ax.set_xlabel("Comm")
    ax.set_ylabel("Latency")
    ax.set_zlabel("Accuracy")
    ax.set_title("3D Tradeoff")
    _save(fig, out_dir / "tradeoff_3d.png")
    plot_count += 1

    categories = ["Acc", "1/Lat", "1/Comm", "Stability", "Activeness"]
    radar_vals = np.array([
        df["accuracy_mean"].mean(),
        1.0 / (df["latency_mean"].mean() + 1e-6),
        1.0 / (df["comm_mean"].mean() + 1e-6),
        1.0 / (df["accuracy_std"].mean() + 1e-6),
        df["active_devices_mean"].mean() / (df["active_devices_mean"].max() + 1e-6),
    ])
    radar_vals = radar_vals / radar_vals.max()
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    vals_closed = np.concatenate([radar_vals, [radar_vals[0]]])
    angles_closed = np.concatenate([angles, [angles[0]]])
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, vals_closed)
    ax.fill(angles_closed, vals_closed, alpha=0.2)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_title("Radar Performance Profile")
    _save(fig, out_dir / "radar_profile.png")
    plot_count += 1

    return plot_count


def generate_comparison_plot(summary_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df))
    ax.bar(x - 0.25, summary_df["best_accuracy"], width=0.25, label="Best Acc")
    ax.bar(x, summary_df["avg_latency"], width=0.25, label="Avg Lat")
    ax.bar(x + 0.25, summary_df["total_communication_mb"], width=0.25, label="Total Comm")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"], rotation=25, ha="right")
    ax.legend()
    ax.set_title("Method Summary Comparison")
    _save(fig, out_path)
