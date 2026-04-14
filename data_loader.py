"""Data loading and distribution utilities."""

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from typing import List, Dict, Tuple
import numpy as np
import os


def get_mnist_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


def load_dataset(dataset_name: str, data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name == "mnist":
        transform = get_mnist_transforms()
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=get_cifar10_transforms(True)
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=get_cifar10_transforms(False)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset


def iid_partition(dataset: Dataset, num_devices: int, seed: int) -> Dict[int, List[int]]:
    """Partition data IID across devices."""
    np.random.seed(seed)
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    
    samples_per_device = num_samples // num_devices
    device_indices = {}
    
    for i in range(num_devices):
        start_idx = i * samples_per_device
        if i == num_devices - 1:
            device_indices[i] = indices[start_idx:].tolist()
        else:
            device_indices[i] = indices[start_idx:start_idx + samples_per_device].tolist()
    
    for i in range(num_devices):
        if len(device_indices[i]) == 0:
            device_indices[i] = [indices[i % len(indices)]]
    
    return device_indices


def dirichlet_partition(
    dataset: Dataset,
    num_devices: int,
    alpha: float,
    seed: int,
    num_classes: int = 10
) -> Dict[int, List[int]]:
    """Partition data non-IID using Dirichlet distribution."""
    np.random.seed(seed)
    
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    
    for c in class_indices:
        np.random.shuffle(class_indices[c])
    
    device_indices = {i: [] for i in range(num_devices)}
    
    for c in range(num_classes):
        indices = class_indices[c]
        proportions = np.random.dirichlet([alpha] * num_devices)
        proportions = proportions / proportions.sum()
        
        splits = (proportions * len(indices)).astype(int)
        splits[-1] = len(indices) - splits[:-1].sum()
        
        current_idx = 0
        for device_id, num_samples in enumerate(splits):
            device_indices[device_id].extend(indices[current_idx:current_idx + num_samples])
            current_idx += num_samples
    
    for i in range(num_devices):
        if len(device_indices[i]) == 0:
            donor = max(device_indices.keys(), key=lambda x: len(device_indices[x]))
            if len(device_indices[donor]) > 1:
                device_indices[i].append(device_indices[donor].pop())
            else:
                device_indices[i] = [0]
    
    return device_indices


def create_data_loaders(
    dataset: Dataset,
    device_indices: Dict[int, List[int]],
    batch_size: int
) -> Dict[int, DataLoader]:
    """Create data loaders for each device."""
    loaders = {}
    for device_id, indices in device_indices.items():
        subset = Subset(dataset, indices)
        loaders[device_id] = DataLoader(
            subset,
            batch_size=min(batch_size, len(indices)),
            shuffle=True,
            drop_last=False
        )
    return loaders


def get_test_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Create test data loader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
