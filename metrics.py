import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List


@dataclass
class RoundMetric:
    round_idx: int
    accuracy: float
    loss: float
    latency: float
    communication_mb: float
    active_devices: int


def write_metrics_csv(path: Path, metrics: List[RoundMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()) if metrics else ["round_idx", "accuracy", "loss", "latency", "communication_mb", "active_devices"])
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))


def summarize_metrics(metrics: List[RoundMetric], method: str, mode: str) -> Dict[str, float]:
    if not metrics:
        return {
            "mode": mode,
            "method": method,
            "best_accuracy": 0.0,
            "avg_latency": 0.0,
            "total_communication_mb": 0.0,
            "convergence_round": 0,
        }

    best = max(metrics, key=lambda x: x.accuracy)
    return {
        "mode": mode,
        "method": method,
        "best_accuracy": best.accuracy,
        "avg_latency": mean(m.latency for m in metrics),
        "total_communication_mb": sum(m.communication_mb for m in metrics),
        "convergence_round": best.round_idx,
    }
