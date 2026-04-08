from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CompressionConfig:
    topk_ratio: float = 0.1
    qsgd_levels: int = 16


@dataclass
class SystemConfig:
    compute_power_range: Tuple[float, float] = (0.5, 2.0)
    bandwidth_range_mbps: Tuple[float, float] = (1.0, 10.0)
    distance_range_m: Tuple[float, float] = (10.0, 100.0)
    clustering_coeff_range: Tuple[float, float] = (0.3, 1.0)
    head_score_weights: Tuple[float, float, float] = (0.45, 0.4, 0.15)  # a, b, c
    quorum_score_weights: Tuple[float, float] = (0.6, 0.4)  # compute, bandwidth
    random_quorum_jitter: float = 0.03

    # latency models
    base_time_per_sample_per_param: float = 3e-10
    head_agg_time_per_param_per_update: float = 1.5e-11
    uav_link_bandwidth_mbps: float = 40.0

    # optional energy model
    energy_compute_coeff: float = 1.2
    energy_tx_coeff: float = 0.8


@dataclass
class TrainConfig:
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 64
    local_epochs: int = 2
    rounds: int = 20


@dataclass
class DataConfig:
    dataset_name: str = "mnist"
    iid: bool = False
    dirichlet_alpha: float = 0.5
    num_classes: int = 10


@dataclass
class ExperimentConfig:
    mode: str = "toy"
    seeds: List[int] = field(default_factory=lambda: [7, 11, 19])
    num_devices: int = 25
    num_clusters: int = 5
    quorum_fraction: float = 0.6
    quorum_count: int = 0
    cluster_formation: str = "distance"
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_root: str = "results"
    methods: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E", "F"])


METHOD_MAP: Dict[str, str] = {
    "A": "standard_fl",
    "B": "clustered_no_compression",
    "C": "clustered_topk_ef",
    "D": "clustered_qsgd",
    "E": "clustered_topk_ef_quorum",
    "F": "clustered_qsgd_quorum",
}


def get_config(mode: str) -> ExperimentConfig:
    if mode not in {"toy", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "toy":
        return ExperimentConfig(
            mode="toy",
            seeds=[7, 11, 19],
            num_devices=25,
            num_clusters=5,
            quorum_fraction=0.6,
            train=TrainConfig(
                optimizer="sgd",
                lr=0.01,
                momentum=0.9,
                batch_size=64,
                local_epochs=2,
                rounds=22,
            ),
            data=DataConfig(dataset_name="mnist", iid=False, dirichlet_alpha=0.5),
        )

    return ExperimentConfig(
        mode="full",
        seeds=[3, 7, 11],
        num_devices=60,
        num_clusters=10,
        quorum_fraction=0.55,
        train=TrainConfig(
            optimizer="sgd",
            lr=0.01,
            momentum=0.9,
            batch_size=64,
            local_epochs=4,
            rounds=60,
        ),
        data=DataConfig(dataset_name="cifar10", iid=False, dirichlet_alpha=0.3),
    )
