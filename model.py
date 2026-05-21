"""
model.py — Lightweight CNNs for MNIST / CIFAR-10. GPU-native.

Public API preserved from the previous MLP version, so federated.py,
compression.py, clustering.py, federated_round.py, etc., do not need to
change:
    get_model, count_parameters, model_size_mb,
    get_flat_params, set_flat_params, clone_model

Architectural choices and why
─────────────────────────────
1. CNNs, not MLP. Priority 1 of the integration brief — MLP on CIFAR is
   the single biggest credibility hit for the paper.

2. GroupNorm, not BatchNorm. Standard convention in FL: BN running
   statistics diverge across heterogeneous clients (Hsieh et al., 2020).
   GN is per-sample and statistically benign under FedAvg.

3. Deliberately *lightweight* (~50 K params MNIST, ~360 K params CIFAR-10).
   ResNet-9 would be ~5 M params and roughly 8–12× slower per round.
   This size is what the brief asks for; ResNet-9 is a drop-in alternative
   (left as a TODO at the bottom).

Input handling
──────────────
Data loaders deliver flat tensors (N, 784) for MNIST and (N, 3072) for
CIFAR. `forward()` reshapes internally so the data pipeline does not need
to change. CIFAR storage layout is channel-major (R…G…B…), so the
(N, 3, 32, 32) reshape is correct.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# MNIST CNN
# ─────────────────────────────────────────────────────────────────────────────

class _MNISTCNN(nn.Module):
    """Conv-Conv-FC-FC. ~50 K params."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.gn1   = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.gn2   = nn.GroupNorm(8, 32)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.gn1(self.conv1(x))), 2)   # 16×14×14
        x = F.max_pool2d(F.relu(self.gn2(self.conv2(x))), 2)   # 32× 7× 7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 CNN
# ─────────────────────────────────────────────────────────────────────────────

class _CIFARCNN(nn.Module):
    """Conv-Conv-Conv-FC-FC with GroupNorm. ~360 K params."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.gn1   = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2   = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn3   = nn.GroupNorm(8, 128)
        self.fc1   = nn.Linear(128 * 4 * 4, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        x = F.max_pool2d(F.relu(self.gn1(self.conv1(x))), 2)   #  32×16×16
        x = F.max_pool2d(F.relu(self.gn2(self.conv2(x))), 2)   #  64× 8× 8
        x = F.max_pool2d(F.relu(self.gn3(self.conv3(x))), 2)   # 128× 4× 4
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper with the public flat-param API
# ─────────────────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    def __init__(self, dataset: str = "MNIST", device: str = "cuda"):
        super().__init__()
        self.dataset = dataset
        self.device_str = device
        self.net = _MNISTCNN() if dataset == "MNIST" else _CIFARCNN()
        self.to(device)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def get_params(self) -> torch.Tensor:
        return torch.cat([p.data.reshape(-1) for p in self.parameters()])

    @torch.no_grad()
    def set_params(self, flat: torch.Tensor) -> None:
        offs = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[offs:offs + n].reshape(p.shape))
            offs += n

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def clone(self) -> "CNN":
        m = CNN(self.dataset, self.device_str)
        m.load_state_dict(self.state_dict())
        return m


# Public API (unchanged signatures)
def get_model(dataset, device="cuda"): return CNN(dataset, device)
def count_parameters(model):           return model.num_params()
def model_size_mb(model, bits=32):     return (model.num_params() * bits) / 8e6
def get_flat_params(model):            return model.get_params()
def set_flat_params(model, flat):      model.set_params(flat)
def clone_model(model):                return model.clone()


# Legacy helpers (kept torch-based for any external import)
def _softmax(x):
    return F.softmax(x, dim=-1)

def _xe(p, y):
    n = y.shape[0]
    return float(-torch.log(p[torch.arange(n, device=p.device), y] + 1e-12).mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# TODO (paper-final): ResNet-9 swap as a drop-in replacement for _CIFARCNN.
# ─────────────────────────────────────────────────────────────────────────────
