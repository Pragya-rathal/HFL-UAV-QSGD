from typing import Dict, List

import pandas as pd



def aggregate_seed_metrics(seed_metrics: List[pd.DataFrame]) -> pd.DataFrame:
    concat = pd.concat(seed_metrics, keys=range(len(seed_metrics)), names=["seed_idx", "row"])
    grouped = concat.groupby("round")
    out = grouped.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        loss_mean=("loss", "mean"),
        loss_std=("loss", "std"),
        latency_mean=("latency", "mean"),
        latency_std=("latency", "std"),
        comm_mean=("comm_mb", "mean"),
        comm_std=("comm_mb", "std"),
        active_devices_mean=("active_devices", "mean"),
        p75_latency_mean=("latency_p75", "mean"),
        max_latency_mean=("latency_max", "mean"),
    ).reset_index()
    return out


def summarize_method(df: pd.DataFrame, method: str, mode: str) -> Dict[str, float]:
    best_accuracy = df["accuracy_mean"].max()
    avg_latency = df["latency_mean"].mean()
    total_comm = df["comm_mean"].sum()
    conv_round = int(df.loc[df["accuracy_mean"].idxmax(), "round"])
    return {
        "mode": mode,
        "method": method,
        "best_accuracy": float(best_accuracy),
        "avg_latency": float(avg_latency),
        "total_communication_mb": float(total_comm),
        "convergence_round": conv_round,
    }
