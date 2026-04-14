"""Metrics tracking and evaluation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field
import json
import os


@dataclass
class RoundMetrics:
    round_num: int
    accuracy: float
    loss: float
    latency: float
    communication_mb: float
    active_devices: int
    cluster_latencies: Optional[Dict[int, float]] = None


@dataclass
class ExperimentMetrics:
    method: str
    rounds: List[RoundMetrics] = field(default_factory=list)
    
    def add_round(self, metrics: RoundMetrics):
        self.rounds.append(metrics)
    
    def get_accuracies(self) -> List[float]:
        return [r.accuracy for r in self.rounds]
    
    def get_losses(self) -> List[float]:
        return [r.loss for r in self.rounds]
    
    def get_latencies(self) -> List[float]:
        return [r.latency for r in self.rounds]
    
    def get_communications(self) -> List[float]:
        return [r.communication_mb for r in self.rounds]
    
    def get_active_devices(self) -> List[int]:
        return [r.active_devices for r in self.rounds]
    
    def best_accuracy(self) -> float:
        return max(self.get_accuracies()) if self.rounds else 0.0
    
    def avg_latency(self) -> float:
        latencies = self.get_latencies()
        return sum(latencies) / len(latencies) if latencies else 0.0
    
    def total_communication(self) -> float:
        return sum(self.get_communications())
    
    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "rounds": [
                {
                    "round": r.round_num,
                    "accuracy": r.accuracy,
                    "loss": r.loss,
                    "latency": r.latency,
                    "communication_mb": r.communication_mb,
                    "active_devices": r.active_devices
                }
                for r in self.rounds
            ],
            "summary": {
                "best_accuracy": self.best_accuracy(),
                "avg_latency": self.avg_latency(),
                "total_communication": self.total_communication()
            }
        }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> tuple:
    """Evaluate model on test set."""
    model.eval()
    model.to(device)
    
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * len(target)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    
    accuracy = 100.0 * correct / total
    avg_loss = test_loss / total
    
    return accuracy, avg_loss


def save_metrics(
    metrics: Dict[str, ExperimentMetrics],
    output_dir: str,
    mode: str
):
    """Save metrics to JSON files."""
    mode_dir = os.path.join(output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    
    for method, exp_metrics in metrics.items():
        filepath = os.path.join(mode_dir, f"method_{method}.json")
        with open(filepath, 'w') as f:
            json.dump(exp_metrics.to_dict(), f, indent=2)
    
    summary_dir = os.path.join(output_dir, mode, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary = {}
    for method, exp_metrics in metrics.items():
        summary[method] = {
            "best_accuracy": exp_metrics.best_accuracy(),
            "avg_latency": exp_metrics.avg_latency(),
            "total_communication": exp_metrics.total_communication()
        }
    
    with open(os.path.join(summary_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)


def load_metrics(output_dir: str, mode: str) -> Dict[str, ExperimentMetrics]:
    """Load metrics from JSON files."""
    mode_dir = os.path.join(output_dir, mode)
    metrics = {}
    
    for filename in os.listdir(mode_dir):
        if filename.startswith("method_") and filename.endswith(".json"):
            method = filename[7:-5]
            filepath = os.path.join(mode_dir, filename)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            exp_metrics = ExperimentMetrics(method=method)
            for r in data["rounds"]:
                exp_metrics.add_round(RoundMetrics(
                    round_num=r["round"],
                    accuracy=r["accuracy"],
                    loss=r["loss"],
                    latency=r["latency"],
                    communication_mb=r["communication_mb"],
                    active_devices=r["active_devices"]
                ))
            metrics[method] = exp_metrics
    
    return metrics
