from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


_SPLIT_CACHE: dict[tuple, dict[int, list[int]]] = {}


def load_dataset(mode: str) -> Dataset:
    mode = mode.lower()
    transform = transforms.ToTensor()

    if mode == "toy":
        return datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    if mode == "full":
        return datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    raise ValueError("mode must be either 'toy' or 'full'")


def _extract_targets(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose targets")

    if isinstance(targets, list):
        return np.asarray(targets, dtype=np.int64)
    if torch.is_tensor(targets):
        return targets.cpu().numpy().astype(np.int64)
    return np.asarray(targets, dtype=np.int64)


def _build_loaders_from_indices(
    dataset: Dataset,
    split_indices: dict[int, list[int]],
    batch_size: int,
) -> dict[int, DataLoader]:
    loaders: dict[int, DataLoader] = {}
    for device_id, indices in split_indices.items():
        subset = Subset(dataset, indices)
        loaders[device_id] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loaders


def split_iid(
    dataset: Dataset,
    num_devices: int,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[int, DataLoader]:
    cache_key = (id(dataset), "iid", num_devices, seed)
    if cache_key not in _SPLIT_CACHE:
        rng = np.random.default_rng(seed)
        all_indices = np.arange(len(dataset))
        rng.shuffle(all_indices)
        shards = np.array_split(all_indices, num_devices)
        _SPLIT_CACHE[cache_key] = {
            device_id: shard.tolist() for device_id, shard in enumerate(shards)
        }

    return _build_loaders_from_indices(dataset, _SPLIT_CACHE[cache_key], batch_size)


def split_dirichlet(
    dataset: Dataset,
    num_devices: int,
    alpha: float,
    seed: int,
    batch_size: int = 32,
) -> dict[int, DataLoader]:
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for Dirichlet split")

    cache_key = (id(dataset), "dirichlet", num_devices, float(alpha), seed)
    if cache_key not in _SPLIT_CACHE:
        rng = np.random.default_rng(seed)
        labels = _extract_targets(dataset)
        num_classes = int(labels.max()) + 1

        class_indices: dict[int, list[int]] = defaultdict(list)
        for idx, cls in enumerate(labels):
            class_indices[int(cls)].append(idx)

        for cls in range(num_classes):
            rng.shuffle(class_indices[cls])

        device_bins: dict[int, list[int]] = {i: [] for i in range(num_devices)}

        # Guarantee at least one sample per device.
        all_indices = np.arange(len(dataset))
        rng.shuffle(all_indices)
        taken = set()
        for device_id in range(num_devices):
            picked_idx = int(all_indices[device_id % len(all_indices)])
            device_bins[device_id].append(picked_idx)
            taken.add(picked_idx)

        # Remove guaranteed samples from class pools.
        for cls in range(num_classes):
            class_indices[cls] = [idx for idx in class_indices[cls] if idx not in taken]

        # Class-wise Dirichlet allocation.
        for cls in range(num_classes):
            cls_pool = class_indices[cls]
            if not cls_pool:
                continue

            proportions = rng.dirichlet(alpha=np.full(num_devices, alpha))
            counts = rng.multinomial(len(cls_pool), proportions)

            start = 0
            for device_id, count in enumerate(counts):
                if count <= 0:
                    continue
                end = start + int(count)
                device_bins[device_id].extend(cls_pool[start:end])
                start = end

        for device_id in range(num_devices):
            rng.shuffle(device_bins[device_id])

        _SPLIT_CACHE[cache_key] = device_bins

    return _build_loaders_from_indices(dataset, _SPLIT_CACHE[cache_key], batch_size)


def build_dataloaders(
    mode: str,
    num_devices: int,
    batch_size: int,
    input_dim: int,
    num_classes: int,
    seed: int,
) -> Tuple[List[DataLoader], DataLoader]:
    del input_dim, num_classes  # Unused with real datasets; kept for compatibility.

    train_dataset = load_dataset(mode)
    transform = transforms.ToTensor()
    if mode.lower() == "toy":
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    else:
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader_dict = split_iid(
        dataset=train_dataset,
        num_devices=num_devices,
        batch_size=batch_size,
        seed=seed,
    )
    device_loaders = [train_loader_dict[i] for i in range(num_devices)]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return device_loaders, test_loader
