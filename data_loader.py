from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


def _build_synthetic_dataset(
    n_samples: int,
    input_dim: int,
    num_classes: int,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, input_dim, generator=generator)
    w = torch.randn(input_dim, num_classes, generator=generator)
    logits = x @ w
    y = logits.argmax(dim=1)
    return TensorDataset(x, y)


def build_dataloaders(
    mode: str,
    num_devices: int,
    batch_size: int,
    input_dim: int,
    num_classes: int,
    seed: int,
) -> Tuple[List[DataLoader], DataLoader]:
    if mode == "toy":
        train_samples = 512
        test_samples = 128
    else:
        train_samples = 4096
        test_samples = 1024

    train_ds = _build_synthetic_dataset(train_samples, input_dim, num_classes, seed)
    test_ds = _build_synthetic_dataset(test_samples, input_dim, num_classes, seed + 1)

    shard_size = len(train_ds) // num_devices
    device_loaders: List[DataLoader] = []

    for i in range(num_devices):
        start = i * shard_size
        end = len(train_ds) if i == num_devices - 1 else (i + 1) * shard_size
        xs = train_ds.tensors[0][start:end]
        ys = train_ds.tensors[1][start:end]
        shard_ds = TensorDataset(xs, ys)
        device_loaders.append(DataLoader(shard_ds, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return device_loaders, test_loader
