from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class FederatedDataManager:
    def __init__(self, dataset_name: str, data_dir: str = "./data"):
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.train_dataset, self.test_dataset, self.in_channels, self.image_size = self._load_dataset()

    def _load_dataset(self):
        if self.dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)
            return train_dataset, test_dataset, 1, 28

        if self.dataset_name == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
            train_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=transform_test)
            return train_dataset, test_dataset, 3, 32

        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def split_indices(
        self,
        num_devices: int,
        iid: bool,
        dirichlet_alpha: float,
        seed: int,
        num_classes: int,
    ) -> Dict[int, np.ndarray]:
        rng = np.random.default_rng(seed)
        labels = np.array(self.train_dataset.targets)
        all_indices = np.arange(len(labels))

        if iid:
            rng.shuffle(all_indices)
            splits = np.array_split(all_indices, num_devices)
            return {i: split for i, split in enumerate(splits)}

        class_indices = [all_indices[labels == c] for c in range(num_classes)]
        for c in range(num_classes):
            rng.shuffle(class_indices[c])

        device_indices: Dict[int, List[int]] = {i: [] for i in range(num_devices)}
        for c in range(num_classes):
            idx_c = class_indices[c]
            proportions = rng.dirichlet([dirichlet_alpha] * num_devices)
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, cuts)
            for d, part in enumerate(splits):
                device_indices[d].extend(part.tolist())

        # ensure every device has data by borrowing if necessary
        all_pool = np.concatenate([np.array(v, dtype=int) for v in device_indices.values() if len(v) > 0])
        for d in range(num_devices):
            if len(device_indices[d]) == 0:
                sampled = rng.choice(all_pool, size=8, replace=False)
                device_indices[d] = sampled.tolist()

        return {d: np.array(idxs, dtype=int) for d, idxs in device_indices.items()}

    def build_device_loaders(
        self, split_map: Dict[int, np.ndarray], batch_size: int, seed: int
    ) -> Dict[int, DataLoader]:
        loaders: Dict[int, DataLoader] = {}
        g = torch.Generator().manual_seed(seed)
        for device_id, indices in split_map.items():
            subset = Subset(self.train_dataset, indices.tolist())
            loaders[device_id] = DataLoader(subset, batch_size=batch_size, shuffle=True, generator=g)
        return loaders

    def build_test_loader(self, batch_size: int = 256) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
