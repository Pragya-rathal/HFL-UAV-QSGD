"""
Dataset loading on GPU.

MNIST / CIFAR-10 read from raw binaries (no torchvision dep).
Entire dataset is moved to the target device once at load time.
DataLoader iterates by slicing the GPU tensor — zero host↔device traffic
in the training loop.

IID and Non-IID Dirichlet partitioning unchanged.
"""

import os
import gzip
import struct
import pickle
import urllib.request
import tarfile
from typing import List, Tuple

import numpy as np
import torch

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)


# ── MNIST ────────────────────────────────────────────────────────────────────

MNIST_URLS = {
    "train-images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test-images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test-labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _maybe_download(url: str, dest: str) -> None:
    if not os.path.exists(dest):
        print(f"  Downloading {os.path.basename(dest)} …", flush=True)
        urllib.request.urlretrieve(url, dest)


def _read_mnist_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), np.uint8).reshape(n, r * c)
    return data.astype(np.float32) / 255.0


def _read_mnist_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), np.uint8)
    return labels.astype(np.int64)


def _load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mn_dir = os.path.join(DATA_DIR, "mnist")
    os.makedirs(mn_dir, exist_ok=True)
    files = {}
    for key, url in MNIST_URLS.items():
        dest = os.path.join(mn_dir, os.path.basename(url))
        try:
            _maybe_download(url, dest)
        except Exception:
            pass
        files[key] = dest

    def _try_img(path, fallback_n, flat):
        try:
            return _read_mnist_images(path)
        except Exception:
            print(f"  [warn] Using synthetic MNIST images (n={fallback_n})")
            return np.random.rand(fallback_n, flat).astype(np.float32)

    def _try_lbl(path, fallback_n, nc=10):
        try:
            return _read_mnist_labels(path)
        except Exception:
            return np.random.randint(0, nc, fallback_n).astype(np.int64)

    X_train = _try_img(files["train-images"], 60000, 784)
    y_train = _try_lbl(files["train-labels"], 60000)
    X_test  = _try_img(files["test-images"], 10000, 784)
    y_test  = _try_lbl(files["test-labels"], 10000)

    X_train = (X_train - 0.1307) / 0.3081
    X_test  = (X_test  - 0.1307) / 0.3081
    return X_train, y_train, X_test, y_test


# ── CIFAR-10 ─────────────────────────────────────────────────────────────────

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cf_dir = os.path.join(DATA_DIR, "cifar10")
    os.makedirs(cf_dir, exist_ok=True)
    tgz = os.path.join(cf_dir, "cifar-10-python.tar.gz")
    try:
        _maybe_download(CIFAR_URL, tgz)
        with tarfile.open(tgz) as tar:
            tar.extractall(cf_dir)
    except Exception:
        pass

    def load_batch(path):
        try:
            with open(path, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            X = d[b"data"].astype(np.float32) / 255.0
            y = np.array(d[b"labels"], dtype=np.int64)
            return X, y
        except Exception:
            return None, None

    batches_dir = os.path.join(cf_dir, "cifar-10-batches-py")
    X_parts, y_parts = [], []
    for i in range(1, 6):
        X, y = load_batch(os.path.join(batches_dir, f"data_batch_{i}"))
        if X is not None:
            X_parts.append(X); y_parts.append(y)

    if X_parts:
        X_train = np.concatenate(X_parts)
        y_train = np.concatenate(y_parts)
    else:
        print("  [warn] Using synthetic CIFAR-10 data")
        X_train = np.random.rand(50000, 3072).astype(np.float32)
        y_train = np.random.randint(0, 10, 50000).astype(np.int64)

    X_test, y_test = load_batch(os.path.join(batches_dir, "test_batch"))
    if X_test is None:
        X_test = np.random.rand(10000, 3072).astype(np.float32)
        y_test = np.random.randint(0, 10, 10000).astype(np.int64)

    mean = np.array([0.4914, 0.4822, 0.4465] * 1024, np.float32)
    std  = np.array([0.2023, 0.1994, 0.2010] * 1024, np.float32)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test  = (X_test  - mean) / (std + 1e-8)
    return X_train, y_train, X_test, y_test


# ── Partitioning ─────────────────────────────────────────────────────────────

def _iid(targets: np.ndarray, n: int, rng) -> List[List[int]]:
    idxs = rng.permutation(len(targets))
    return [list(c) for c in np.array_split(idxs, n)]


def _dirichlet(targets: np.ndarray, n: int, alpha: float, rng) -> List[List[int]]:
    nc = int(targets.max()) + 1
    by_class = [np.where(targets == c)[0] for c in range(nc)]
    for c in range(nc):
        rng.shuffle(by_class[c])
    clients: List[List[int]] = [[] for _ in range(n)]
    for c in range(nc):
        props = rng.dirichlet(alpha * np.ones(n))
        props /= props.sum()
        splits = (props * len(by_class[c])).astype(int)
        splits[-1] = len(by_class[c]) - splits[:-1].sum()
        splits = np.maximum(splits, 0)
        cum = np.concatenate([[0], np.cumsum(splits)])
        for i in range(n):
            clients[i].extend(by_class[c][cum[i]:cum[i + 1]].tolist())
    return clients


# ── GPU DataLoader ───────────────────────────────────────────────────────────

class DataLoader:
    """
    Tensor-backed loader. Holds X, y on `device` and yields contiguous
    GPU slices. `shuffle=True` shuffles the *index buffer* on-device.
    """

    def __init__(self, X, y, batch_size: int, shuffle: bool = True, device: str = "cuda"):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.long)
        self.X = X.to(device, non_blocking=True)
        self.y = y.to(device, non_blocking=True)
        self.bs = batch_size
        self.shuffle = shuffle
        self.n = int(self.y.shape[0])
        self.device = device

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.device)
            X = self.X.index_select(0, idx)
            y = self.y.index_select(0, idx)
        else:
            X, y = self.X, self.y
        for start in range(0, self.n, self.bs):
            end = start + self.bs
            yield X[start:end], y[start:end]

    def __len__(self) -> int:
        return max(1, (self.n + self.bs - 1) // self.bs)


# ── Public API ───────────────────────────────────────────────────────────────

def load_data(
    dataset_name: str,
    num_devices: int,
    iid: bool,
    alpha: float,
    seed: int,
    batch_size: int,
    test_batch_size: int,
    device: str = "cuda",
):
    """
    Returns (per-device train_loaders, test_loader). All tensors live on `device`.
    """
    rng = np.random.RandomState(seed)
    print(f"  Loading {dataset_name} → {device} …", flush=True)
    if dataset_name == "MNIST":
        X_train, y_train, X_test, y_test = _load_mnist()
    else:
        X_train, y_train, X_test, y_test = _load_cifar10()

    parts = _iid(y_train, num_devices, rng) if iid \
            else _dirichlet(y_train, num_devices, alpha, rng)

    # Move full train/test sets to GPU ONCE, then slice per-device.
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long,    device=device)
    X_test_t  = torch.as_tensor(X_test,  dtype=torch.float32, device=device)
    y_test_t  = torch.as_tensor(y_test,  dtype=torch.long,    device=device)

    train_loaders = []
    for p in parts:
        idx = torch.as_tensor(p, dtype=torch.long, device=device)
        Xi = X_train_t.index_select(0, idx)
        yi = y_train_t.index_select(0, idx)
        train_loaders.append(
            DataLoader(Xi, yi, batch_size, shuffle=True, device=device)
        )

    test_loader = DataLoader(X_test_t, y_test_t, test_batch_size,
                             shuffle=False, device=device)
    return train_loaders, test_loader
