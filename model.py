"""
PyTorch MLP for MNIST / CIFAR-10 (2 hidden layers) — GPU-native.

Public API (preserved from NumPy version):
    get_model, count_parameters, model_size_mb,
    get_flat_params, set_flat_params, clone_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MNIST  : 784  → 256 → 128 → 10
    CIFAR10: 3072 → 512 → 256 → 10
    """

    def __init__(self, dataset: str = "MNIST", device: str = "cuda"):
        super().__init__()
        self.dataset = dataset
        self.device_str = device

        if dataset == "MNIST":
            d0, d1, d2, d3 = 784, 256, 128, 10
        else:
            d0, d1, d2, d3 = 3072, 512, 256, 10

        self.fc1 = nn.Linear(d0, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.fc3 = nn.Linear(d2, d3)

        # He init (matches the original _he initialiser)
        for m in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # ── Flat-param helpers (preserve interface) ─────────────────────────────
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

    def clone(self) -> "MLP":
        m = MLP(self.dataset, self.device_str)
        m.load_state_dict(self.state_dict())
        return m


# ── Public helpers ───────────────────────────────────────────────────────────

def get_model(dataset: str, device: str = "cuda") -> MLP:
    return MLP(dataset, device)

def count_parameters(model: MLP) -> int:
    return model.num_params()

def model_size_mb(model: MLP, bits: int = 32) -> float:
    return (model.num_params() * bits) / 8e6

def get_flat_params(model: MLP) -> torch.Tensor:
    return model.get_params()

def set_flat_params(model: MLP, flat: torch.Tensor) -> None:
    model.set_params(flat)

def clone_model(model: MLP) -> MLP:
    return model.clone()


# ── Legacy soft-max / cross-entropy helpers ──────────────────────────────────
# Kept torch-based for any external code that imports them.

def _softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)

def _xe(p: torch.Tensor, y: torch.Tensor) -> float:
    n = y.shape[0]
    return float(-torch.log(p[torch.arange(n, device=p.device), y] + 1e-12).mean().item())
