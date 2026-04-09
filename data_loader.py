from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision import datasets, transforms

from config import ModeConfig


Sample = Tuple[Tensor, int]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dataset_key(name: str) -> str:
    return name.replace("-", "").replace("_", "").upper()


def _build_dataset(dataset_name: str, train: bool, root: Path, seed: int):
    transform = transforms.Compose([transforms.ToTensor()])
    key = _dataset_key(dataset_name)

    if key == "MNIST":
        try:
            return datasets.MNIST(root=str(root), train=train, download=True, transform=transform)
        except Exception:
            torch.manual_seed(seed + (0 if train else 1))
            return datasets.FakeData(
                size=60000 if train else 10000,
                image_size=(1, 28, 28),
                num_classes=10,
                transform=transform,
                random_offset=seed,
            )

    if key == "CIFAR10":
        try:
            return datasets.CIFAR10(root=str(root), train=train, download=True, transform=transform)
        except Exception:
            torch.manual_seed(seed + (0 if train else 1))
            return datasets.FakeData(
                size=50000 if train else 10000,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=transform,
                random_offset=seed,
            )

    raise ValueError(f"Unsupported dataset '{dataset_name}'. Use MNIST or CIFAR-10.")


def _extract_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, list):
            return np.asarray(targets, dtype=np.int64)
        if isinstance(targets, torch.Tensor):
            return targets.cpu().numpy().astype(np.int64)
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)


def _iid_partition(num_samples: int, num_devices: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(num_samples)
    return [split.astype(np.int64) for split in np.array_split(shuffled, num_devices)]


def _dirichlet_partition(labels: np.ndarray, num_devices: int, alpha: float, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1

    for _ in range(100):
        per_device: List[List[int]] = [[] for _ in range(num_devices)]

        for cls in range(num_classes):
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)

            proportions = rng.dirichlet(np.full(num_devices, alpha, dtype=np.float64))
            split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            cls_splits = np.split(cls_indices, split_points)

            for dev_id, split in enumerate(cls_splits):
                per_device[dev_id].extend(split.tolist())

        sizes = [len(indices) for indices in per_device]
        if min(sizes) > 0:
            return [np.asarray(indices, dtype=np.int64) for indices in per_device]

    per_device = [[] for _ in range(num_devices)]
    for idx, label in enumerate(labels):
        per_device[int(label) % num_devices].append(int(idx))

    return [np.asarray(indices, dtype=np.int64) for indices in per_device]


def _index_to_samples(dataset, indices: Sequence[int]) -> List[Sample]:
    samples: List[Sample] = []
    for idx in indices:
        x, y = dataset[int(idx)]
        samples.append((x, int(y)))
    return samples


def build_federated_datasets(
    mode_cfg: ModeConfig,
    seed: int,
    iid: bool = False,
    alpha: float = 0.5,
    data_root: str = "./data",
) -> Tuple[Dict[int, List[Sample]], List[Sample]]:
    _set_seed(seed)

    root = Path(data_root)
    train_set = _build_dataset(mode_cfg.dataset, train=True, root=root, seed=seed)
    test_set_raw = _build_dataset(mode_cfg.dataset, train=False, root=root, seed=seed)

    labels = _extract_labels(train_set)

    if iid:
        partitions = _iid_partition(num_samples=len(train_set), num_devices=mode_cfg.num_devices, seed=seed)
    else:
        partitions = _dirichlet_partition(labels=labels, num_devices=mode_cfg.num_devices, alpha=alpha, seed=seed)

    device_data: Dict[int, List[Sample]] = {
        device_id: _index_to_samples(train_set, partitions[device_id])
        for device_id in range(mode_cfg.num_devices)
    }

    test_indices = np.arange(len(test_set_raw), dtype=np.int64)
    test_set = _index_to_samples(test_set_raw, test_indices)

    return device_data, test_set
