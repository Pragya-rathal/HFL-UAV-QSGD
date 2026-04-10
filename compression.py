"""
Gradient compression: Top-K + Error Feedback, QSGD.
Pure NumPy.
"""

import numpy as np
from typing import Tuple


# ── Top-K + Error Feedback ────────────────────────────────────────────────────

def topk_compress(update, residual, fraction):
    """
    u = update + residual
    top-k of |u|; new_residual = u - compressed
    Returns (values, indices, new_residual)
    """
    d = update.size
    k = max(1, int(d * fraction))
    u = update + residual
    topk_idx = np.argpartition(np.abs(u), -k)[-k:]
    values    = u[topk_idx].copy()
    new_res   = u.copy(); new_res[topk_idx] = 0.0
    return values, topk_idx, new_res


def topk_decompress(values, indices, total_size):
    dense = np.zeros(total_size, np.float32)
    dense[indices] = values
    return dense


def topk_message_size_mb(k, bits=32):
    return (k * bits + k * 32) / 8e6


# ── QSGD ─────────────────────────────────────────────────────────────────────

def qsgd_quantize(update, levels):
    """Approx unbiased stochastic quantisation (Alistarh et al. 2017)."""
    norm = float(np.linalg.norm(update))
    if norm < 1e-12:
        return np.zeros_like(update, np.int8), 1.0
    scale = norm / levels
    norm_u = update / scale
    floors = np.floor(np.abs(norm_u))
    frac   = np.abs(norm_u) - floors
    rand   = np.random.rand(*update.shape).astype(np.float32)
    rounded = floors + (rand < frac).astype(np.float32)
    rounded = np.clip(rounded, 0, levels)
    signs   = np.sign(update); signs[signs == 0] = 1.0
    quantised = np.clip((signs * rounded).astype(np.int8), -127, 127)
    return quantised, scale


def qsgd_dequantize(quantised, scale):
    return quantised.astype(np.float32) * scale


def qsgd_message_size_mb(d, levels, bits=32):
    bpe = int(np.ceil(np.log2(levels + 1))) + 1
    return (d * bpe + bits) / 8e6


# ── Full-model size ───────────────────────────────────────────────────────────

def full_update_size_mb(num_params, bits=32):
    return (num_params * bits) / 8e6
