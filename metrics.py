### metrics.py
import numpy as np
import json
import os


def compute_final_metrics(history):
    """Compute final summary metrics from round history."""
    if not history:
        return {}

    accuracies = [h['accuracy'] for h in history]
    latencies = [h['latency'] for h in history]
    comms = [h['communication_MB'] for h in history]
    losses = [h['loss'] for h in history]

    return {
        'best_accuracy': float(max(accuracies)),
        'final_accuracy': float(accuracies[-1]),
        'average_latency': float(np.mean(latencies)),
        'total_latency': float(np.sum(latencies)),
        'total_communication_MB': float(np.sum(comms)),
        'average_communication_MB': float(np.mean(comms)),
        'final_loss': float(losses[-1]),
        'min_loss': float(min(losses)),
        'num_rounds': len(history),
    }


def save_results(history, final_metrics, method_name, mode, results_dir='results'):
    """Save results to JSON files."""
    mode_dir = os.path.join(results_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)

    # Per-round history
    history_path = os.path.join(mode_dir, f'{method_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Summary metrics
    summaries_dir = os.path.join(results_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    summary_path = os.path.join(summaries_dir, f'{mode}_{method_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    return history_path, summary_path


def print_summary(method_name, final_metrics):
    print(f"\n{'='*50}")
    print(f"  Summary: {method_name}")
    print(f"{'='*50}")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print(f"{'='*50}\n")


def compare_methods(all_results):
    """Print comparison table of all methods."""
    print("\n" + "=" * 80)
    print(f"{'Method':<10} {'BestAcc':>10} {'FinalAcc':>10} {'AvgLat(s)':>12} {'TotalComm(MB)':>15}")
    print("-" * 80)
    for name, metrics in all_results.items():
        print(f"{name:<10} {metrics['best_accuracy']:>10.4f} {metrics['final_accuracy']:>10.4f} "
              f"{metrics['average_latency']:>12.4f} {metrics['total_communication_MB']:>15.2f}")
    print("=" * 80)
