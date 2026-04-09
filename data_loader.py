import random
from typing import Dict, List, Tuple

from config import ModeConfig


Sample = Tuple[List[float], int]


def _make_sample(feature_dim: int, label: int, rng: random.Random) -> Sample:
    features = [rng.random() for _ in range(feature_dim)]
    return features, label


def build_federated_datasets(mode_cfg: ModeConfig, seed: int, iid: bool = False) -> Tuple[Dict[int, List[Sample]], List[Sample]]:
    rng = random.Random(seed)
    feature_dim = 16 if mode_cfg.dataset == "MNIST" else 32
    num_classes = 10
    samples_per_device = 40 if mode_cfg.name == "toy" else 60

    device_data: Dict[int, List[Sample]] = {}
    for device_id in range(mode_cfg.num_devices):
        local: List[Sample] = []
        for idx in range(samples_per_device):
            if iid:
                label = rng.randrange(num_classes)
            else:
                label = (device_id + idx) % num_classes
            local.append(_make_sample(feature_dim, label, rng))
        device_data[device_id] = local

    test_set: List[Sample] = []
    test_size = 400 if mode_cfg.name == "toy" else 1000
    for _ in range(test_size):
        label = rng.randrange(num_classes)
        test_set.append(_make_sample(feature_dim, label, rng))

    return device_data, test_set
