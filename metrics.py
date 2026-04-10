"""
Metrics aggregation, summary statistics, and convergence analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


METHOD_LABELS = {
    "standard_fl": "A: Std-FL",
    "clustered_fl": "B: Clustered-FL",
    "topk_ef": "C: Top-K+EF",
    "qsgd": "D: QSGD",
    "topk_quorum": "E: Top-K+Quorum (Prop.)",
    "qsgd_quorum": "F: QSGD+Quorum (Prop.)",
}


def history_to_df(history: List[Dict], method: str, seed: int) -> pd.DataFrame:
    rows = []
    for h in history:
        row = dict(h)
        row["method"] = method
        row["seed"] = seed
        row.pop("cluster_times", None)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_seeds(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Given per-seed DataFrames, compute mean ± std across seeds per round."""
    combined = pd.concat(dfs)
    numeric_cols = [c for c in combined.columns
                    if c not in ("method", "seed", "round")]
    agg = combined.groupby("round")[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                   for c in agg.columns]
    return agg


def compute_summary(all_dfs: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    """
    Per-method summary: best_acc, avg_latency, total_comm, convergence_round.
    mean ± std across seeds.
    """
    rows = []
    for method, dfs in all_dfs.items():
        method_stats = []
        for df in dfs:
            best_acc = df["accuracy"].max()
            avg_lat = df["latency_round"].mean()
            total_comm = df["comm_total_mb"].sum()
            # Convergence: first round where accuracy >= 0.95 * best_acc
            threshold = 0.95 * best_acc
            conv_rounds = df[df["accuracy"] >= threshold]["round"]
            conv_round = int(conv_rounds.iloc[0]) if len(conv_rounds) > 0 else len(df)
            method_stats.append((best_acc, avg_lat, total_comm, conv_round))

        arr = np.array(method_stats)
        rows.append({
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "best_acc_mean": arr[:, 0].mean(),
            "best_acc_std": arr[:, 0].std(),
            "avg_latency_mean": arr[:, 1].mean(),
            "avg_latency_std": arr[:, 1].std(),
            "total_comm_mb_mean": arr[:, 2].mean(),
            "total_comm_mb_std": arr[:, 2].std(),
            "convergence_round_mean": arr[:, 3].mean(),
            "convergence_round_std": arr[:, 3].std(),
        })

    return pd.DataFrame(rows)


def print_summary_table(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("METHOD COMPARISON TABLE")
    print("=" * 90)
    hdr = (f"{'Method':<30} {'Best Acc':>10} {'Avg Lat(s)':>12} "
           f"{'Total Comm(MB)':>16} {'Conv Round':>12}")
    print(hdr)
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(
            f"{row['label']:<30} "
            f"{row['best_acc_mean']:6.4f}±{row['best_acc_std']:.4f}  "
            f"{row['avg_latency_mean']:8.3f}±{row['avg_latency_std']:.3f}  "
            f"{row['total_comm_mb_mean']:10.2f}±{row['total_comm_mb_std']:.2f}  "
            f"{row['convergence_round_mean']:6.1f}±{row['convergence_round_std']:.1f}"
        )
    print("=" * 90)

    # Best method under each constraint
    print("\nBEST METHOD UNDER CONSTRAINTS:")
    best_acc_idx = summary_df["best_acc_mean"].idxmax()
    best_lat_idx = summary_df["avg_latency_mean"].idxmin()
    best_comm_idx = summary_df["total_comm_mb_mean"].idxmin()

    print(f"  Accuracy  constraint → {summary_df.loc[best_acc_idx, 'label']}"
          f"  (acc={summary_df.loc[best_acc_idx, 'best_acc_mean']:.4f})")
    print(f"  Latency   constraint → {summary_df.loc[best_lat_idx, 'label']}"
          f"  (lat={summary_df.loc[best_lat_idx, 'avg_latency_mean']:.3f}s)")
    print(f"  Comm.     constraint → {summary_df.loc[best_comm_idx, 'label']}"
          f"  (comm={summary_df.loc[best_comm_idx, 'total_comm_mb_mean']:.2f}MB)")
    print()


def get_cluster_latency_stats(history_list: List[List[Dict]]) -> pd.DataFrame:
    """Extract per-cluster latency distribution across rounds and seeds."""
    rows = []
    for history in history_list:
        for h in history:
            for ct in h.get("cluster_times", []):
                rows.append({"round": h["round"], "cluster_latency": ct})
    return pd.DataFrame(rows)
