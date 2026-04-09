from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModeConfig:
    name: str
    dataset: str
    num_devices: int
    num_clusters: int
    rounds: int
    local_epochs: int


@dataclass(frozen=True)
class GlobalConfig:
    seed: int
    methods: List[str]
    modes: Dict[str, ModeConfig]


def build_config(seed: int = 42) -> GlobalConfig:
    modes = {
        "toy": ModeConfig(
            name="toy",
            dataset="MNIST",
            num_devices=20,
            num_clusters=4,
            rounds=20,
            local_epochs=2,
        ),
        "full": ModeConfig(
            name="full",
            dataset="CIFAR10",
            num_devices=50,
            num_clusters=8,
            rounds=50,
            local_epochs=3,
        ),
    }
    methods = [
        "standard_fl",
        "clustered_fl",
        "cluster_topk_ef",
        "cluster_qsgd",
        "cluster_topk_ef_quorum",
        "cluster_qsgd_quorum",
    ]
    return GlobalConfig(seed=seed, methods=methods, modes=modes)
