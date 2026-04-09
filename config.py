from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModeConfig:
    dataset: str
    num_devices: int
    num_clusters: int
    rounds: int
    local_epochs: int
    batch_size: int
    lr: float
    momentum: float
    dirichlet_alpha: float
    seeds: List[int]
    samples_per_device: int
    test_subset: int


@dataclass
class MethodConfig:
    name: str
    clustered: bool
    compression: str  # none | topk | qsgd
    quorum: bool


@dataclass
class SimulationConfig:
    mode: str
    iid: bool = False
    cluster_head_weights: Dict[str, float] = field(default_factory=lambda: {
        "compute": 0.45,
        "clustering": 0.4,
        "bandwidth": 0.15,
    })
    topk_ratio: float = 0.02
    qsgd_levels: int = 16
    quorum_fraction: float = 0.6
    device_base_compute_time: float = 0.25
    head_aggregation_time: float = 0.02
    uav_bandwidth_mbps: float = 25.0
    device_to_head_distance_max: float = 200.0
    mode_config: ModeConfig = None  # type: ignore[assignment]


MODES: Dict[str, ModeConfig] = {
    "toy": ModeConfig(
        dataset="MNIST",
        num_devices=24,
        num_clusters=5,
        rounds=22,
        local_epochs=2,
        batch_size=64,
        lr=0.02,
        momentum=0.9,
        dirichlet_alpha=0.5,
        seeds=[11, 29, 47],
        samples_per_device=160,
        test_subset=4000,
    ),
    "full": ModeConfig(
        dataset="CIFAR10",
        num_devices=50,
        num_clusters=8,
        rounds=50,
        local_epochs=3,
        batch_size=64,
        lr=0.03,
        momentum=0.9,
        dirichlet_alpha=0.3,
        seeds=[13, 31, 59],
        samples_per_device=120,
        test_subset=8000,
    ),
}


METHODS: List[MethodConfig] = [
    MethodConfig("standard_fl", clustered=False, compression="none", quorum=False),
    MethodConfig("clustered_fl", clustered=True, compression="none", quorum=False),
    MethodConfig("cluster_topk_ef", clustered=True, compression="topk", quorum=False),
    MethodConfig("cluster_qsgd", clustered=True, compression="qsgd", quorum=False),
    MethodConfig("cluster_topk_ef_quorum", clustered=True, compression="topk", quorum=True),
    MethodConfig("cluster_qsgd_quorum", clustered=True, compression="qsgd", quorum=True),
]


def build_config(mode: str, iid: bool = False) -> SimulationConfig:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode {mode}")
    sim_config = SimulationConfig(mode=mode, iid=iid)
    sim_config.mode_config = MODES[mode]
    return sim_config
