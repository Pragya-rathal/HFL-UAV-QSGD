from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from config import ModeConfig


def _load_dataset(name: str, train: bool, root: str = "./data") -> Dataset:
    if name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        return datasets.MNIST(root=root, train=train, transform=transform, download=True)
    if name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        return datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
    raise ValueError(f"Unsupported dataset: {name}")


def _labels_array(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        arr = dataset.targets
        if isinstance(arr, list):
            return np.array(arr)
        return np.array(arr)
    raise ValueError("Dataset does not expose targets")


def dirichlet_split(
    labels: np.ndarray,
    num_devices: int,
    alpha: float,
    samples_per_device: int,
    rng: np.random.Generator,
) -> Dict[int, List[int]]:
    classes = np.unique(labels)
    class_indices = {c: np.where(labels == c)[0].tolist() for c in classes}
    for c in classes:
        rng.shuffle(class_indices[c])

    device_indices = defaultdict(list)
    for c in classes:
        idxs = class_indices[c]
        proportions = rng.dirichlet(np.repeat(alpha, num_devices))
        splits = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        chunks = np.split(np.array(idxs), splits)
        for d, chunk in enumerate(chunks):
            device_indices[d].extend(chunk.tolist())

    for d in range(num_devices):
        rng.shuffle(device_indices[d])
        device_indices[d] = device_indices[d][:samples_per_device]

    return dict(device_indices)


def iid_split(
    total_indices: np.ndarray,
    num_devices: int,
    samples_per_device: int,
    rng: np.random.Generator,
) -> Dict[int, List[int]]:
    shuffled = total_indices.copy()
    rng.shuffle(shuffled)
    needed = num_devices * samples_per_device
    trimmed = shuffled[:needed]
    return {
        d: trimmed[d * samples_per_device : (d + 1) * samples_per_device].tolist()
        for d in range(num_devices)
    }


def build_federated_dataloaders(
    mode_cfg: ModeConfig,
    iid: bool,
    seed: int,
) -> Tuple[Dict[int, DataLoader], DataLoader, Dict[int, List[int]]]:
    rng = np.random.default_rng(seed)
    train_ds = _load_dataset(mode_cfg.dataset, train=True)
    test_ds = _load_dataset(mode_cfg.dataset, train=False)

    labels = _labels_array(train_ds)
    if iid:
        partitions = iid_split(
            np.arange(len(train_ds)),
            mode_cfg.num_devices,
            mode_cfg.samples_per_device,
            rng,
        )
    else:
        partitions = dirichlet_split(
            labels,
            mode_cfg.num_devices,
            mode_cfg.dirichlet_alpha,
            mode_cfg.samples_per_device,
            rng,
        )

    device_loaders: Dict[int, DataLoader] = {}
    for d, idxs in partitions.items():
        subset = Subset(train_ds, idxs)
        device_loaders[d] = DataLoader(subset, batch_size=mode_cfg.batch_size, shuffle=True)

    test_idxs = np.arange(len(test_ds))
    rng.shuffle(test_idxs)
    test_subset = Subset(test_ds, test_idxs[: mode_cfg.test_subset].tolist())
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)

    return device_loaders, test_loader, partitions


def infer_channels(dataset_name: str) -> int:
    return 1 if dataset_name == "MNIST" else 3
