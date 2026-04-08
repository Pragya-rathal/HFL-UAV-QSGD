from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RoundMetric:
    round_id: int
    seed: int
    method: str
    train_loss: float
    test_accuracy: float
    comm_mb: float
    latency_s: float
    active_devices: int
    energy_j: float
    p75_latency_s: float
    max_latency_s: float
    compression_ratio: float


def metrics_to_df(metrics: List[RoundMetric]) -> pd.DataFrame:
    return pd.DataFrame([m.__dict__ for m in metrics])


def convergence_round(acc_series: pd.Series, target: float = 0.9) -> Optional[int]:
    hits = acc_series[acc_series >= target]
    if len(hits) == 0:
        return None
    return int(hits.index[0]) + 1


def summarize_method(df: pd.DataFrame, method_name: str) -> Dict:
    final = df.sort_values(["seed", "round_id"]).groupby("seed").tail(1)
    per_seed = final[["test_accuracy", "latency_s", "comm_mb", "energy_j"]]

    return {
        "method": method_name,
        "final_acc_mean": per_seed["test_accuracy"].mean(),
        "final_acc_std": per_seed["test_accuracy"].std(ddof=0),
        "avg_latency_mean": df.groupby("seed")["latency_s"].mean().mean(),
        "avg_latency_std": df.groupby("seed")["latency_s"].mean().std(ddof=0),
        "total_comm_mean": df.groupby("seed")["comm_mb"].sum().mean(),
        "total_comm_std": df.groupby("seed")["comm_mb"].sum().std(ddof=0),
        "energy_mean": df.groupby("seed")["energy_j"].sum().mean(),
        "best_acc_mean": df.groupby("seed")["test_accuracy"].max().mean(),
        "efficiency_score": float(per_seed["test_accuracy"].mean() / (df["latency_s"].mean() + 1e-9)),
    }


def paired_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    s = summary_df.set_index("method")
    rows = []
    if "standard_fl" in s.index and "clustered_no_compression" in s.index:
        rows.append({
            "comparison": "Standard FL vs Clustered FL",
            "acc_delta": s.loc["clustered_no_compression", "final_acc_mean"] - s.loc["standard_fl", "final_acc_mean"],
            "latency_delta": s.loc["clustered_no_compression", "avg_latency_mean"] - s.loc["standard_fl", "avg_latency_mean"],
            "comm_delta": s.loc["clustered_no_compression", "total_comm_mean"] - s.loc["standard_fl", "total_comm_mean"],
        })
    if "clustered_no_compression" in s.index and "clustered_topk_ef" in s.index:
        rows.append({
            "comparison": "Clustered FL vs TopK+EF",
            "acc_delta": s.loc["clustered_topk_ef", "final_acc_mean"] - s.loc["clustered_no_compression", "final_acc_mean"],
            "latency_delta": s.loc["clustered_topk_ef", "avg_latency_mean"] - s.loc["clustered_no_compression", "avg_latency_mean"],
            "comm_delta": s.loc["clustered_topk_ef", "total_comm_mean"] - s.loc["clustered_no_compression", "total_comm_mean"],
        })
    if "clustered_no_compression" in s.index and "clustered_topk_ef_quorum" in s.index:
        rows.append({
            "comparison": "Clustered FL vs TopK+EF+Quorum",
            "acc_delta": s.loc["clustered_topk_ef_quorum", "final_acc_mean"] - s.loc["clustered_no_compression", "final_acc_mean"],
            "latency_delta": s.loc["clustered_topk_ef_quorum", "avg_latency_mean"] - s.loc["clustered_no_compression", "avg_latency_mean"],
            "comm_delta": s.loc["clustered_topk_ef_quorum", "total_comm_mean"] - s.loc["clustered_no_compression", "total_comm_mean"],
        })
    return pd.DataFrame(rows)
