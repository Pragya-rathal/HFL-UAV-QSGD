### main.py
import os
import sys
import time
import torch

from config import get_config, set_seed, parse_args
from data_loader import prepare_data
from devices import create_devices
from federated import FederatedTrainer
from metrics import compute_final_metrics, save_results, print_summary, compare_methods
from plotting import generate_all_plots


def run_experiment(config):
    mode = config['mode']
    print(f"\n{'='*60}")
    print(f"  Hierarchical Federated Learning for UAV-IoT Networks")
    print(f"  Mode: {mode.upper()} | Dataset: {config['dataset'].upper()}")
    print(f"  Devices: {config['num_devices']} | Clusters: {config['num_clusters']}")
    print(f"  Rounds: {config['num_rounds']} | Distribution: {config['data_distribution']}")
    print(f"{'='*60}\n")

    set_seed(config['seed'])

    # Prepare data
    print("Loading and partitioning data...")
    device_loaders, test_loader, dataset_sizes = prepare_data(config)
    print(f"  Data ready. Sizes range: [{min(dataset_sizes)}, {max(dataset_sizes)}]")

    # Create devices
    print("Creating IoT devices...")
    devices = create_devices(config, device_loaders, dataset_sizes)
    print(f"  {len(devices)} devices created.")

    all_histories = {}
    all_summaries = {}

    methods = [
        ('A_Standard_FL',   'run_standard_fl'),
        ('B_Clustered_FL',  'run_clustered_fl'),
        ('C_TopK_EF',       'run_clustered_topk'),
        ('D_QSGD',          'run_clustered_qsgd'),
        ('E_TopK_Quorum',   'run_clustered_topk_quorum'),
        ('F_QSGD_Quorum',   'run_clustered_qsgd_quorum'),
    ]

    for method_name, method_fn in methods:
        print(f"\n{'─'*60}")
        print(f"  Running Method: {method_name}")
        print(f"{'─'*60}")

        # Fresh trainer for each method (same global init seed)
        set_seed(config['seed'])
        trainer = FederatedTrainer(config, devices, test_loader)

        t0 = time.time()
        run_fn = getattr(trainer, method_fn)
        history = run_fn(config['num_rounds'])
        elapsed = time.time() - t0

        final_metrics = compute_final_metrics(history)
        final_metrics['wall_clock_seconds'] = round(elapsed, 2)

        save_results(history, final_metrics, method_name, mode)
        print_summary(method_name, final_metrics)

        all_histories[method_name] = history
        all_summaries[method_name] = final_metrics

    # Compare all methods
    compare_methods(all_summaries)

    # Generate plots
    generate_all_plots(all_histories, all_summaries, mode)

    print(f"\nExperiment complete. Results saved to results/{mode}/")
    return all_histories, all_summaries


def main():
    args = parse_args()
    config = get_config(args.mode)

    # Device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing compute device: {device}")

    # Run
    run_experiment(config)


if __name__ == '__main__':
    main()
