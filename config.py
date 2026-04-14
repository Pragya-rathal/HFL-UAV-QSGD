"""Configuration for Hierarchical Federated Learning experiments."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeviceConfig:
    compute_power_range: tuple = (0.5, 2.0)
    bandwidth_range: tuple = (1.0, 10.0)
    distance_range: tuple = (10.0, 100.0)
    clustering_coeff_range: tuple = (0.3, 1.0)


@dataclass
class DataConfig:
    dataset: str = "mnist"
    iid: bool = True
    dirichlet_alpha: float = 0.5
    batch_size: int = 32
    test_batch_size: int = 128


@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    local_epochs: int = 2
    num_rounds: int = 25
    optimizer: str = "sgd"


@dataclass
class CompressionConfig:
    topk_ratio: float = 0.1
    qsgd_levels: int = 8


@dataclass
class QuorumConfig:
    fraction: float = 0.6
    rotation_window: int = 5
    compute_weight: float = 0.7
    bandwidth_weight: float = 0.3
    noise_std: float = 0.1


@dataclass
class ClusteringConfig:
    num_clusters: int = 5
    d0: float = 50.0
    cc_weight: float = 0.5
    compute_weight: float = 0.3
    bandwidth_weight: float = 0.2


@dataclass
class ExperimentConfig:
    mode: str = "toy"
    seed: int = 42
    num_devices: int = 25
    device: DeviceConfig = field(default_factory=DeviceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    quorum: QuorumConfig = field(default_factory=QuorumConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    results_dir: str = "results"
    methods: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E", "F"])


def get_toy_config() -> ExperimentConfig:
    config = ExperimentConfig(
        mode="toy",
        seed=42,
        num_devices=25,
    )
    config.data.dataset = "mnist"
    config.data.batch_size = 32
    config.training.num_rounds = 25
    config.training.local_epochs = 2
    config.clustering.num_clusters = 5
    return config


def get_full_config() -> ExperimentConfig:
    config = ExperimentConfig(
        mode="full",
        seed=42,
        num_devices=75,
    )
    config.data.dataset = "cifar10"
    config.data.batch_size = 32
    config.training.num_rounds = 75
    config.training.local_epochs = 4
    config.clustering.num_clusters = 10
    return config


def get_config(mode: str) -> ExperimentConfig:
    if mode == "toy":
        return get_toy_config()
    elif mode == "full":
        return get_full_config()
    else:
        raise ValueError(f"Unknown mode: {mode}")
