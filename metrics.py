from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def evaluate(model, dataloader, criterion) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            total_acc += accuracy(logits, yb)
            total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_batches,
        "accuracy": total_acc / total_batches,
    }


class MetricsTracker:
    """Tracks round metrics and exports CSV per method."""

    def __init__(self, method: str):
        self.method = method
        self.rounds: List[Dict[str, float]] = []

    def log_round(
        self,
        round_idx: int,
        accuracy_value: float,
        loss_value: float,
        latency_value: float,
        communication_value: float,
        active_devices_value: float,
    ) -> None:
        self.rounds.append(
            {
                "round": float(round_idx),
                "accuracy": float(accuracy_value),
                "loss": float(loss_value),
                "latency": float(latency_value),
                "communication": float(communication_value),
                "active_devices": float(active_devices_value),
            }
        )

    def final_metrics(self) -> Dict[str, float]:
        if not self.rounds:
            return {
                "avg_latency": 0.0,
                "total_communication": 0.0,
                "best_accuracy": 0.0,
            }

        avg_latency = sum(r["latency"] for r in self.rounds) / len(self.rounds)
        total_communication = sum(r["communication"] for r in self.rounds)
        best_accuracy = max(r["accuracy"] for r in self.rounds)

        return {
            "avg_latency": float(avg_latency),
            "total_communication": float(total_communication),
            "best_accuracy": float(best_accuracy),
        }

    def save_csv(self, output_dir: str = ".") -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        safe_method = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.method)
        csv_path = output_path / f"metrics_{safe_method}.csv"

        fieldnames = ["round", "accuracy", "loss", "latency", "communication", "active_devices"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rounds:
                writer.writerow(row)

        return str(csv_path)


def save_history_csv(history: Dict[str, List[float]], method: str, output_dir: str = ".") -> str:
    """
    Save per-round metrics from a history dict and return path.

    Expected keys:
      - accuracy
      - loss
      - latency
      - communication
      - active_devices
    """
    tracker = MetricsTracker(method=method)

    rounds = max(
        len(history.get("accuracy", [])),
        len(history.get("loss", [])),
        len(history.get("latency", [])),
        len(history.get("communication", [])),
        len(history.get("active_devices", [])),
    )

    for i in range(rounds):
        tracker.log_round(
            round_idx=i,
            accuracy_value=history.get("accuracy", [0.0] * rounds)[i]
            if i < len(history.get("accuracy", []))
            else 0.0,
            loss_value=history.get("loss", [0.0] * rounds)[i]
            if i < len(history.get("loss", []))
            else 0.0,
            latency_value=history.get("latency", [0.0] * rounds)[i]
            if i < len(history.get("latency", []))
            else 0.0,
            communication_value=history.get("communication", [0.0] * rounds)[i]
            if i < len(history.get("communication", []))
            else 0.0,
            active_devices_value=history.get("active_devices", [0.0] * rounds)[i]
            if i < len(history.get("active_devices", []))
            else 0.0,
        )

    return tracker.save_csv(output_dir=output_dir)
