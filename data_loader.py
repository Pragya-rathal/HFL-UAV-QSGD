### data_loader.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


def get_transforms(dataset_name):
    if dataset_name == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:  # cifar10
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])


def load_dataset(dataset_name, data_dir='./data'):
    transform = get_transforms(dataset_name)
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def iid_partition(dataset, num_devices, seed=42):
    rng = np.random.RandomState(seed)
    num_samples = len(dataset)
    indices = rng.permutation(num_samples)
    device_indices = np.array_split(indices, num_devices)
    return [idx.tolist() for idx in device_indices]


def non_iid_dirichlet_partition(dataset, num_devices, alpha=0.5, seed=42):
    rng = np.random.RandomState(seed)
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    device_indices = [[] for _ in range(num_devices)]
    for c in range(num_classes):
        proportions = rng.dirichlet(alpha * np.ones(num_devices))
        c_idx = class_indices[c]
        rng.shuffle(c_idx)
        splits = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        splits_idx = np.split(c_idx, splits)
        for d in range(num_devices):
            device_indices[d].extend(splits_idx[d].tolist())
    # Guarantee every device has at least 1 sample
    for d in range(num_devices):
        if len(device_indices[d]) == 0:
            # steal one sample from the device with the most
            donor = max(range(num_devices), key=lambda x: len(device_indices[x]))
            device_indices[d].append(device_indices[donor].pop())
    return device_indices


def create_device_loaders(train_dataset, device_indices, batch_size=32):
    loaders = []
    for idx in device_indices:
        if len(idx) == 0:
            idx = [0]  # fallback
        subset = Subset(train_dataset, idx)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)
        loaders.append(loader)
    return loaders


def create_test_loader(test_dataset, batch_size=256):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def prepare_data(config):
    train_dataset, test_dataset = load_dataset(config['dataset'])
    num_devices = config['num_devices']
    seed = config['seed']

    if config['data_distribution'] == 'iid':
        device_indices = iid_partition(train_dataset, num_devices, seed=seed)
    else:
        device_indices = non_iid_dirichlet_partition(
            train_dataset, num_devices,
            alpha=config['dirichlet_alpha'], seed=seed
        )

    device_loaders = create_device_loaders(train_dataset, device_indices, config['batch_size'])
    test_loader = create_test_loader(test_dataset)
    dataset_sizes = [len(idx) for idx in device_indices]
    return device_loaders, test_loader, dataset_sizes
