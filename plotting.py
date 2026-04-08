import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sns.set_theme(style="whitegrid", context="talk")


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def generate_all_plots(
    all_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_root: str,
    sensitivity: Dict[str, pd.DataFrame],
) -> int:
    base = os.path.join(output_root, "plots")
    count = 0

    grouped = all_df.groupby(["method", "round_id"]).agg(
        acc_mean=("test_accuracy", "mean"),
        acc_std=("test_accuracy", "std"),
        loss_mean=("train_loss", "mean"),
        lat_mean=("latency_s", "mean"),
        comm_mean=("comm_mb", "mean"),
        active_mean=("active_devices", "mean"),
        comp_mean=("compression_ratio", "mean"),
    ).reset_index()

    # 1-5 core
    for y, name, folder, ylabel in [
        ("acc_mean", "accuracy_vs_rounds", "core", "Accuracy"),
        ("loss_mean", "loss_vs_rounds", "core", "Training Loss"),
        ("lat_mean", "latency_vs_rounds", "core", "Latency (s)"),
        ("comm_mean", "communication_vs_rounds", "core", "Communication (MB)"),
        ("active_mean", "active_clients_vs_rounds", "core", "Active Devices"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for m, g in grouped.groupby("method"):
            ax.plot(g["round_id"], g[y], label=m)
            if y == "acc_mean":
                std = g["acc_std"].fillna(0)
                ax.fill_between(g["round_id"], g[y] - std, g[y] + std, alpha=0.2)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        _save(fig, os.path.join(base, folder, f"{name}.png"))
        count += 1

    # 6-9 final bars
    for col, name in [
        ("final_acc_mean", "final_accuracy_bar"),
        ("avg_latency_mean", "avg_latency_bar"),
        ("total_comm_mean", "total_comm_bar"),
        ("energy_mean", "energy_bar"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        err = summary_df[col.replace("_mean", "_std")] if col.replace("_mean", "_std") in summary_df.columns else None
        ax.bar(summary_df["method"], summary_df[col], yerr=err)
        ax.set_xticklabels(summary_df["method"], rotation=30, ha="right")
        ax.set_ylabel(col)
        _save(fig, os.path.join(base, "core", f"{name}.png"))
        count += 1

    # 10-12 tradeoff scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(summary_df["avg_latency_mean"], summary_df["final_acc_mean"])
    for _, r in summary_df.iterrows():
        ax.annotate(r["method"], (r["avg_latency_mean"], r["final_acc_mean"]))
    ax.set_xlabel("Avg Latency (s)")
    ax.set_ylabel("Final Accuracy")
    _save(fig, os.path.join(base, "tradeoffs", "accuracy_vs_latency.png")); count += 1

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(summary_df["total_comm_mean"], summary_df["final_acc_mean"])
    for _, r in summary_df.iterrows():
        ax.annotate(r["method"], (r["total_comm_mean"], r["final_acc_mean"]))
    ax.set_xlabel("Total Communication (MB)")
    ax.set_ylabel("Final Accuracy")
    _save(fig, os.path.join(base, "tradeoffs", "accuracy_vs_communication.png")); count += 1

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(summary_df["total_comm_mean"], summary_df["avg_latency_mean"])
    for _, r in summary_df.iterrows():
        ax.annotate(r["method"], (r["total_comm_mean"], r["avg_latency_mean"]))
    ax.set_xlabel("Total Communication (MB)")
    ax.set_ylabel("Avg Latency (s)")
    _save(fig, os.path.join(base, "tradeoffs", "communication_vs_latency.png")); count += 1

    # 13-16 efficiency
    e_df = summary_df.copy()
    base_acc = e_df.set_index("method").loc["standard_fl", "final_acc_mean"] if "standard_fl" in set(e_df["method"]) else 1.0
    base_comm = e_df.set_index("method").loc["standard_fl", "total_comm_mean"] if "standard_fl" in set(e_df["method"]) else 1.0
    base_lat = e_df.set_index("method").loc["standard_fl", "avg_latency_mean"] if "standard_fl" in set(e_df["method"]) else 1.0
    e_df["acc_per_mb"] = e_df["final_acc_mean"] / e_df["total_comm_mean"].clip(lower=1e-9)
    e_df["acc_per_sec"] = e_df["final_acc_mean"] / e_df["avg_latency_mean"].clip(lower=1e-9)
    e_df["comm_reduction_ratio"] = 1 - e_df["total_comm_mean"] / base_comm
    e_df["lat_reduction_ratio"] = 1 - e_df["avg_latency_mean"] / base_lat
    for col, fn in [
        ("acc_per_mb", "accuracy_per_mb"),
        ("acc_per_sec", "accuracy_per_second"),
        ("comm_reduction_ratio", "communication_reduction_ratio"),
        ("lat_reduction_ratio", "latency_reduction_ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(e_df["method"], e_df[col])
        ax.set_xticklabels(e_df["method"], rotation=30, ha="right")
        _save(fig, os.path.join(base, "tradeoffs", f"{fn}.png"))
        count += 1

    # 17 latency boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=all_df, x="method", y="latency_s", ax=ax)
    ax.tick_params(axis="x", rotation=30)
    _save(fig, os.path.join(base, "latency", "latency_distribution_per_method.png")); count += 1

    # 18 per-round histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=all_df, x="latency_s", hue="method", ax=ax, element="step", stat="density", common_norm=False)
    _save(fig, os.path.join(base, "latency", "per_round_latency_histogram.png")); count += 1

    # 19 cluster-wise proxy using p75/max
    fig, ax = plt.subplots(figsize=(10, 6))
    tmp = all_df.groupby(["method", "round_id"])["p75_latency_s"].mean().reset_index()
    for m, g in tmp.groupby("method"):
        ax.plot(g["round_id"], g["p75_latency_s"], label=m)
    ax.legend(fontsize=8)
    _save(fig, os.path.join(base, "latency", "cluster_wise_latency_breakdown.png")); count += 1

    # 20 device-level latency approximation from per-round per-device avg
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=all_df, x="method", y="max_latency_s", ax=ax)
    ax.tick_params(axis="x", rotation=30)
    _save(fig, os.path.join(base, "latency", "device_level_latency_distribution.png")); count += 1

    # 21 participation
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, g in all_df.groupby("method"):
        gx = g.groupby("round_id")["active_devices"].mean().reset_index()
        ax.plot(gx["round_id"], gx["active_devices"], label=m)
    ax.legend(fontsize=8)
    _save(fig, os.path.join(base, "participation", "participating_devices_per_round.png")); count += 1

    # 22-24 from sensitivity data
    for key, fn in [("quorum_acc", "quorum_size_vs_accuracy"), ("quorum_lat", "quorum_size_vs_latency"), ("participation_freq", "participation_frequency_per_device")]:
        if key in sensitivity:
            fig, ax = plt.subplots(figsize=(9, 6))
            sdf = sensitivity[key]
            if key == "participation_freq":
                sns.histplot(data=sdf, x="participation_count", hue="method", ax=ax, bins=20)
            else:
                sns.lineplot(data=sdf, x="x", y="y", hue="method", marker="o", ax=ax)
            _save(fig, os.path.join(base, "participation", f"{fn}.png")); count += 1

    # 25-28 compression
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, g in grouped.groupby("method"):
        ax.plot(g["round_id"], g["comp_mean"], label=m)
    ax.legend(fontsize=8)
    _save(fig, os.path.join(base, "compression", "compression_ratio_vs_rounds.png")); count += 1

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=all_df.groupby("method", as_index=False).agg(acc=("test_accuracy", "mean"), comp=("compression_ratio", "mean")), x="comp", y="acc", hue="method", ax=ax)
    _save(fig, os.path.join(base, "compression", "compression_ratio_vs_accuracy.png")); count += 1

    for key, fn in [("topk_perf", "sparsity_vs_performance"), ("qsgd_perf", "quantization_levels_vs_performance")]:
        if key in sensitivity:
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.lineplot(data=sensitivity[key], x="x", y="y", marker="o", hue="method", ax=ax)
            _save(fig, os.path.join(base, "compression", f"{fn}.png")); count += 1

    # 29-32 scaling
    for key, fn in [
        ("scale_dev_lat", "num_devices_vs_latency"),
        ("scale_dev_comm", "num_devices_vs_communication"),
        ("scale_cluster_lat", "cluster_size_vs_latency"),
        ("scale_cluster_acc", "cluster_size_vs_accuracy"),
    ]:
        if key in sensitivity:
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.lineplot(data=sensitivity[key], x="x", y="y", hue="method", marker="o", ax=ax)
            _save(fig, os.path.join(base, "scaling", f"{fn}.png")); count += 1

    # 33-39 robustness/convergence
    for key, fn, folder in [
        ("acc_var", "accuracy_variance_across_seeds", "robustness"),
        ("lat_var", "latency_variance_across_seeds", "robustness"),
        ("alpha_sens", "sensitivity_noniid_alpha", "robustness"),
        ("bw_sens", "sensitivity_bandwidth", "robustness"),
        ("round_target", "rounds_to_target_accuracy", "robustness"),
        ("conv_speed", "convergence_speed_comparison", "robustness"),
        ("early_late", "early_late_performance_gap", "robustness"),
    ]:
        if key in sensitivity:
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.barplot(data=sensitivity[key], x="x", y="y", hue="method", ax=ax)
            _save(fig, os.path.join(base, folder, f"{fn}.png")); count += 1

    # 40 3D scatter
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(summary_df["final_acc_mean"], summary_df["avg_latency_mean"], summary_df["total_comm_mean"])
    for _, r in summary_df.iterrows():
        ax.text(r["final_acc_mean"], r["avg_latency_mean"], r["total_comm_mean"], r["method"], fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Latency")
    ax.set_zlabel("Communication")
    _save(fig, os.path.join(base, "tradeoffs", "3d_accuracy_latency_communication.png")); count += 1

    # 41 radar chart
    radar = summary_df.copy()
    metrics = ["final_acc_mean", "avg_latency_mean", "total_comm_mean", "efficiency_score"]
    norm = radar[metrics].copy()
    norm["avg_latency_mean"] = 1.0 / (norm["avg_latency_mean"] + 1e-9)
    norm["total_comm_mean"] = 1.0 / (norm["total_comm_mean"] + 1e-9)
    for m in metrics:
        norm[m] = norm[m] / norm[m].max()

    import numpy as np

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, row in norm.iterrows():
        vals = row[metrics].tolist() + [row[metrics].tolist()[0]]
        ax.plot(angles, vals, label=radar.iloc[i]["method"])
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Accuracy", "Latency(inv)", "Comm(inv)", "Efficiency"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    _save(fig, os.path.join(base, "tradeoffs", "radar_method_comparison.png")); count += 1

    return count
