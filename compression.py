"""
Gradient compression on GPU.

Top-K + Error Feedback, QSGD. All ops on torch tensors — input device
determines output device. Public API matches the NumPy version.
"""

import math
import torch


# ── Top-K + Error Feedback ───────────────────────────────────────────────────

def topk_compress(update: torch.Tensor, residual: torch.Tensor, fraction: float):
    """
    u = update + residual; pick top-k of |u|; new_residual = u with those positions zeroed.

    Returns (values, indices, new_residual) — all torch tensors on update.device.
    """
    d = update.numel()
    k = max(1, int(d * fraction))
    u = update + residual
    _, topk_idx = torch.topk(u.abs(), k)
    values = u[topk_idx].clone()
    new_res = u.clone()
    new_res[topk_idx] = 0.0
    return values, topk_idx, new_res


def topk_decompress(values: torch.Tensor, indices: torch.Tensor, total_size: int) -> torch.Tensor:
    dense = torch.zeros(total_size, dtype=values.dtype, device=values.device)
    dense[indices] = values
    return dense


def topk_message_size_mb(k: int, bits: int = 32) -> float:
    # k values + k indices (32-bit each)
    return (k * bits + k * 32) / 8e6


# ── QSGD ─────────────────────────────────────────────────────────────────────

def qsgd_quantize(update: torch.Tensor, levels: int):
    """
    Approx unbiased stochastic quantisation (Alistarh et al. 2017).
    Returns (quantised int8 tensor, scale float).
    """
    norm = update.norm()
    if norm.item() < 1e-12:
        return torch.zeros_like(update, dtype=torch.int8), 1.0

    scale = norm / levels
    norm_u = update / scale
    abs_u = norm_u.abs()
    floors = abs_u.floor()
    frac = abs_u - floors
    rand = torch.rand_like(frac)
    rounded = (floors + (rand < frac).to(frac.dtype)).clamp_(0, levels)

    signs = torch.sign(update)
    signs[signs == 0] = 1.0

    quantised = (signs * rounded).to(torch.int8).clamp_(-127, 127)
    return quantised, scale.item()


def qsgd_dequantize(quantised: torch.Tensor, scale: float) -> torch.Tensor:
    return quantised.to(torch.float32) * scale


def qsgd_message_size_mb(d: int, levels: int, bits: int = 32) -> float:
    bpe = int(math.ceil(math.log2(levels + 1))) + 1  # bits per element
    return (d * bpe + bits) / 8e6


# ── Full-model size ──────────────────────────────────────────────────────────

def full_update_size_mb(num_params: int, bits: int = 32) -> float:
    return (num_params * bits) / 8e6
