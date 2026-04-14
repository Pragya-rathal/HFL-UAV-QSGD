"""Plotting utilities for experiment results."""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List
from metrics import ExperimentMetrics


METHOD_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
    "F": "#8c564b"
}

METHOD_LABELS = {
    "A": "Standard FL",
    "B": "Clustered FL",
    "C": "Cluster + Top-K",
    "D": "Cluster + QSGD",
    "E": "Cluster + Top-K + Quorum",
    "F": "Cluster + QSGD + Quorum"
}


def setup_plot_style():
    """Set up consistent plot style."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })


def plot_accuracy_vs_rounds(
    metrics: Dict[str, ExperimentMetrics],
    output_dir: str,
    mode: str
):
    """Plot accuracy vs communication rounds."""
    setup_plot_style()
    fig, ax = plt.subplots()
    
    for method, exp_metrics in sorted(metrics.items()):
        accuracies = exp_metrics.get_accuracies()
        rounds = list(range(1, len(accuracies) + 1))
        ax.plot(
            rounds,
            accuracies,
            color=METHOD_COLORS.get(method, "gray"),
            label=METHOD_LABELS.get(method, method),
            marker='o',
            markevery=max(1, len(rounds) // 10)
        )
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Accuracy vs Communication Rounds ({mode.upper()} mode)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, mode, "accuracy_vs_rounds.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_vs_rounds(
    metrics: Dict[str, ExperimentMetrics],
    output_dir: str,
    mode: str
):
    """Plot loss vs communication rounds."""
    setup_plot_style()
    fig, ax = plt.subplots()
    
    for method, exp_metrics in sorted(metrics.items()):
        losses = exp_metrics.get_losses()
        rounds = list(range(1, len(losses) + 1))
        ax.plot(
            rounds,
            losses,
            color=METHOD_COLORS.get(method, "gray"),
            label=METHOD_LABELS.get(method, method),
            marker='s',
            markevery=max(1, len(rounds) // 10)
        )
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Loss")
    ax.set_title(f"Loss vs Communication Rounds ({mode.upper()} mode)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, mode, "loss_vs_rounds.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_latency_vs_rounds(
    metrics: Dict[str, ExperimentMetrics],
    output_dir: str,
    mode: str
):
    """Plot latency vs communication rounds."""
    setup_plot_style()
    fig, ax = plt.subplots()
    
    for method, exp_metrics in sorted(metrics.items()):
        latencies = exp_metrics.get_latencies()
        rounds = list(range(1, len(latencies) + 1))
        ax.plot(
            rounds,
            latencies,
            color=METHOD_COLORS.get(method, "gray"),
            label=METHOD_LABELS.get(method, method),
            marker='^',
            markevery=max(1, len(rounds) // 10)
        )
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Round Latency (s)")
    ax.set_title(f"Latency vs Communication Rounds ({mode.upper()} mode)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, mode, "latency_vs_rounds.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_communication_vs_rounds(
    metrics: Dict[str, ExperimentMetrics],
    output_dir: str,
    mode: str
):
    """Plot cumulative communication vs rounds."""
    setup_plot_style()
    fig, ax = plt.subplots()
    
    for method, exp_metrics in sorted(metrics.items()):
        communications = exp_metrics.get_communications()
        cumulative = np.cumsum(communications)
        rounds = list(range(1, len(cumulative) + 1))
        ax.plot(
            rounds,
            cumulative,
            color=METHOD_COLORS.get(method, "gray"),
            label=METHOD_LABELS.get(method, method),
            marker='d',
            markevery=max(1, len(rounds) // 10)
        )
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Cumulative Communication (MB)")
    ax.set_title(f"Cumulative Communication vs Rounds ({mode.upper()} mode)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, mode, "communication_vs_rounds.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tradeoff_accuracy_communication(
    metrics: Dict[str, ExperimentMetrics],
    output_dir: str,
    mode: str
):
    """Plot accuracy vs total communication trade-off."""
    setup_plot_style()
    fig, ax = plt.subplots()
    
    for method, exp_metrics in sorted(metrics.items()):
        best_acc = exp_metrics.best_accuracy()
        total_comm = exp_metrics.total_communication()
        ax.scatter(
            total_comm,
            best_acc,
            color=METHOD_COLORS.get(method, "gray"),
            label=METHOD_LABELS.get(method, method),
            s=150,
            marker='o',
            edgecolors='black',
            linewidths=1.5
        )
    
    ax.set_xlabel("Total Communication (MB)")
    ax.set_ylabel("Best Accuracy (%)")
    ax.set_title(f"Accuracy vs Communication Trade-off ({mode.upper()} mode)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, mode, "tradeoff_accuracy_communication.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tradeoff_accuracy_latency(
    metrics: Dict[str, ExperimentMetrics],
    output_dir: str,
    mode: str
):
    """Plot accuracy vs average latency trade-off."""
    setup_plot_style()
    fig, ax = plt.subplots()
    
    for method, exp_metrics in sorted(metrics.items()):
        best_acc = exp_metrics.best_accuracy()
        avg_lat = exp_metrics.avg_latency()
        ax.scatter(
            avg_lat,
            best_acc,
            color=METHOD_COLORS.get(method, "gray"),
            label=METHOD_LABELS.get(method, method),
            s=150,
            marker='s',
            edgecolors='black',
            linewidths=1.5
        )
    
    ax.set_xlabel("Average Latency (s)")
    ax.set_ylabel("Best Accuracy (%)")
    ax.set_title(f"Accuracy vs Latency Trade-off ({mode.upper()} mode)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, mode, "tradeoff_accuracy_latency.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_bar_chart(
    metrics: Dict[str, ExperimentMetrics],
    output_dir: str,
    mode: str
):
    """Plot summary bar chart comparing all methods."""
    setup_plot_style()
    
    methods = sorted(metrics.keys())
    x = np.arange(len(methods))
    width = 0.25
    
    best_accs = [metrics[m].best_accuracy() for m in methods]
    avg_lats = [metrics[m].avg_latency() for m in methods]
    total_comms = [metrics[m].total_communication() for m in methods]
    
    max_lat = max(avg_lats) if max(avg_lats) > 0 else 1
    max_comm = max(total_comms) if max(total_comms) > 0 else 1
    
    norm_lats = [l / max_lat * 100 for l in avg_lats]
    norm_comms = [c / max_comm * 100 for c in total_comms]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, best_accs, width, label='Best Accuracy (%)', color='#2ecc71')
    bars2 = ax.bar(x, norm_lats, width, label='Norm. Avg Latency', color='#e74c3c')
    bars3 = ax.bar(x + width, norm_comms, width, label='Norm. Total Comm.', color='#3498db')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Value (normalized to 100)')
    ax.set_title(f'Method Comparison Summary ({mode.upper()} mode)')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, mode, "summary_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_plots(
    metrics: Dict[str, ExperimentMetrics],
    output_dir: str,
    mode: str
):
    """Generate all plots for experiment results."""
    os.makedirs(os.path.join(output_dir, mode), exist_ok=True)
    
    plot_accuracy_vs_rounds(metrics, output_dir, mode)
    plot_loss_vs_rounds(metrics, output_dir, mode)
    plot_latency_vs_rounds(metrics, output_dir, mode)
    plot_communication_vs_rounds(metrics, output_dir, mode)
    plot_tradeoff_accuracy_communication(metrics, output_dir, mode)
    plot_tradeoff_accuracy_latency(metrics, output_dir, mode)
    plot_summary_bar_chart(metrics, output_dir, mode)
    
    print(f"All plots saved to {os.path.join(output_dir, mode)}/")
