"""Main entry point for hierarchical federated learning experiments."""

import argparse
import os
import sys
import torch
import numpy as np
import copy
from typing import Dict

from config import get_config, ExperimentConfig
from data_loader import (
    load_dataset,
    iid_partition,
    dirichlet_partition,
    create_data_loaders,
    get_test_loader
)
from model import get_model, flatten_model, count_parameters
from devices import create_devices, Device
from clustering import cluster_devices, get_cluster_info
from federated import FederatedTrainer
from metrics import ExperimentMetrics, RoundMetrics, save_metrics
from plotting import generate_all_plots


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    method: str,
    trainer: FederatedTrainer,
    initial_model: torch.nn.Module,
    config: ExperimentConfig
) -> ExperimentMetrics:
    """Run a single experiment for one method."""
    print(f"\n{'='*60}")
    print(f"Running Method {method}: {get_method_description(method)}")
    print(f"{'='*60}")
    
    trainer.reset(initial_model)
    
    exp_metrics = ExperimentMetrics(method=method)
    
    for round_num in range(1, config.training.num_rounds + 1):
        if method == "A":
            round_metrics = trainer.train_round_standard(
                round_num,
                config.training.local_epochs
            )
        elif method == "B":
            round_metrics = trainer.train_round_clustered(
                round_num,
                config.training.local_epochs
            )
        elif method == "C":
            round_metrics = trainer.train_round_topk(
                round_num,
                config.training.local_epochs
            )
        elif method == "D":
            round_metrics = trainer.train_round_qsgd(
                round_num,
                config.training.local_epochs
            )
        elif method == "E":
            round_metrics = trainer.train_round_topk_quorum(
                round_num,
                config.training.local_epochs
            )
        elif method == "F":
            round_metrics = trainer.train_round_qsgd_quorum(
                round_num,
                config.training.local_epochs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        exp_metrics.add_round(round_metrics)
        
        if round_num % 5 == 0 or round_num == 1:
            print(f"  Round {round_num:3d}: "
                  f"Acc={round_metrics.accuracy:.2f}%, "
                  f"Loss={round_metrics.loss:.4f}, "
                  f"Lat={round_metrics.latency:.3f}s, "
                  f"Comm={round_metrics.communication_mb:.2f}MB, "
                  f"Active={round_metrics.active_devices}")
    
    print(f"\n  Final Results:")
    print(f"    Best Accuracy: {exp_metrics.best_accuracy():.2f}%")
    print(f"    Avg Latency: {exp_metrics.avg_latency():.3f}s")
    print(f"    Total Communication: {exp_metrics.total_communication():.2f}MB")
    
    return exp_metrics


def get_method_description(method: str) -> str:
    """Get human-readable method description."""
    descriptions = {
        "A": "Standard FL (no clustering)",
        "B": "Clustered FL",
        "C": "Clustered FL + Top-K + Error Feedback",
        "D": "Clustered FL + QSGD",
        "E": "Clustered FL + Top-K + Quorum",
        "F": "Clustered FL + QSGD + Quorum"
    }
    return descriptions.get(method, "Unknown")


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Federated Learning in UAV-assisted IoT Networks"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["toy", "full"],
        default="toy",
        help="Experiment mode: 'toy' for quick testing, 'full' for complete experiment"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Specific methods to run (e.g., A B C). Default: all methods"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    config = get_config(args.mode)
    config.seed = args.seed
    config.results_dir = args.output_dir
    
    if args.methods:
        config.methods = args.methods
    
    set_seed(config.seed)
    
    print("="*60)
    print(f"Hierarchical Federated Learning Experiment")
    print(f"Mode: {config.mode.upper()}")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {config.data.dataset.upper()}")
    print(f"  Num Devices: {config.num_devices}")
    print(f"  Num Clusters: {config.clustering.num_clusters}")
    print(f"  Num Rounds: {config.training.num_rounds}")
    print(f"  Local Epochs: {config.training.local_epochs}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Batch Size: {config.data.batch_size}")
    print(f"  Methods: {config.methods}")
    print(f"  Seed: {config.seed}")
    
    print("\nLoading dataset...")
    train_dataset, test_dataset = load_dataset(config.data.dataset)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    print("\nPartitioning data...")
    if config.data.iid:
        device_indices = iid_partition(
            train_dataset,
            config.num_devices,
            config.seed
        )
        print("  Distribution: IID")
    else:
        device_indices = dirichlet_partition(
            train_dataset,
            config.num_devices,
            config.data.dirichlet_alpha,
            config.seed
        )
        print(f"  Distribution: Non-IID (Dirichlet α={config.data.dirichlet_alpha})")
    
    samples_per_device = [len(indices) for indices in device_indices.values()]
    print(f"  Samples per device: min={min(samples_per_device)}, "
          f"max={max(samples_per_device)}, avg={np.mean(samples_per_device):.1f}")
    
    data_loaders = create_data_loaders(
        train_dataset,
        device_indices,
        config.data.batch_size
    )
    test_loader = get_test_loader(test_dataset, config.data.test_batch_size)
    
    print("\nInitializing model...")
    model = get_model(config.data.dataset)
    num_params = count_parameters(model)
    print(f"  Model: {type(model).__name__}")
    print(f"  Parameters: {num_params:,}")
    
    print("\nCreating devices...")
    devices = create_devices(
        config.num_devices,
        data_loaders,
        model,
        config,
        config.seed
    )
    
    compute_powers = [d.properties.compute_power for d in devices]
    bandwidths = [d.properties.bandwidth for d in devices]
    print(f"  Compute Power: min={min(compute_powers):.2f}, "
          f"max={max(compute_powers):.2f}, avg={np.mean(compute_powers):.2f}")
    print(f"  Bandwidth: min={min(bandwidths):.2f}, "
          f"max={max(bandwidths):.2f}, avg={np.mean(bandwidths):.2f} Mbps")
    
    print("\nClustering devices...")
    clusters, cluster_heads, cc_values, apl_values = cluster_devices(
        devices,
        config.clustering.num_clusters,
        config.clustering.d0,
        config.clustering.cc_weight,
        config.clustering.compute_weight,
        config.clustering.bandwidth_weight,
        config.seed
    )
    
    cluster_info = get_cluster_info(clusters, cluster_heads, devices)
    print(f"  Clusters formed: {cluster_info['num_clusters']}")
    print(f"  Cluster sizes: {cluster_info['cluster_sizes']}")
    print(f"  Cluster heads: {cluster_info['cluster_heads']}")
    
    print("\nInitializing trainer...")
    initial_model = copy.deepcopy(model)
    trainer = FederatedTrainer(
        model=model,
        devices=devices,
        test_loader=test_loader,
        config=config,
        clusters=clusters,
        cluster_heads=cluster_heads
    )
    
    all_metrics: Dict[str, ExperimentMetrics] = {}
    
    for method in config.methods:
        exp_metrics = run_experiment(method, trainer, initial_model, config)
        all_metrics[method] = exp_metrics
    
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    os.makedirs(config.results_dir, exist_ok=True)
    save_metrics(all_metrics, config.results_dir, config.mode)
    print(f"  Metrics saved to {config.results_dir}/{config.mode}/")
    
    print("\nGenerating plots...")
    generate_all_plots(all_metrics, config.results_dir, config.mode)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n{'Method':<35} {'Best Acc':<12} {'Avg Lat':<12} {'Total Comm':<15}")
    print("-"*74)
    for method in sorted(all_metrics.keys()):
        m = all_metrics[method]
        desc = get_method_description(method)
        print(f"{method}: {desc:<30} {m.best_accuracy():>8.2f}%    "
              f"{m.avg_latency():>8.3f}s    {m.total_communication():>10.2f}MB")
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
