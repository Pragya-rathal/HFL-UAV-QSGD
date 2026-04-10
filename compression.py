from __future__ import annotations

import math

import torch


def _as_flat_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.float32).reshape(-1)


def topk_error_feedback(
    update: torch.Tensor,
    residual: torch.Tensor | None,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Top-K with error feedback.

    u = update + residual
    k = max(1, int(ratio * len(u)))
    compressed[topk_indices] = u[topk_indices]
    residual = u - compressed

    Size model:
      values:  k * 32 bits
      indices: k * 32 bits
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1].")

    flat_update = _as_flat_float_tensor(update)

    if residual is None:
        flat_residual = torch.zeros_like(flat_update)
    else:
        flat_residual = _as_flat_float_tensor(residual)
        if flat_residual.numel() != flat_update.numel():
            raise ValueError("residual must have the same flattened size as update.")

    u = flat_update + flat_residual
    k = max(1, int(ratio * u.numel()))

    topk_indices = torch.topk(u.abs(), k=k, largest=True, sorted=False).indices
    compressed = torch.zeros_like(u)
    compressed[topk_indices] = u[topk_indices]

    new_residual = u - compressed

    total_bits = (k * 32) + (k * 32)
    size_mb = total_bits / (8.0 * 1024.0 * 1024.0)

    return compressed, new_residual, size_mb


def qsgd_compress(update: torch.Tensor, s: int = 256) -> tuple[torch.Tensor, float]:
    """
    QSGD compression (dequantized reconstruction output).

    - Normalize vector
    - Quantize to s levels
    - Stochastic rounding
    - Dequantize

    Size model: log2(s) bits per parameter
    """
    if s < 2:
        raise ValueError("s must be >= 2.")

    v = _as_flat_float_tensor(update)
    n = v.numel()
    bits_per_param = math.log2(float(s))

    if n == 0:
        return v, 0.0

    norm = torch.linalg.vector_norm(v, ord=2)
    if torch.isclose(norm, torch.tensor(0.0, dtype=v.dtype, device=v.device)):
        size_mb = (n * bits_per_param) / (8.0 * 1024.0 * 1024.0)
        return torch.zeros_like(v), size_mb

    scaled = v.abs() * (float(s) / norm)
    lower = torch.floor(scaled)
    lower = torch.clamp(lower, min=0.0, max=float(s - 1))

    prob = scaled - lower
    rand = torch.rand_like(prob)
    levels = lower + (rand < prob).to(v.dtype)
    levels = torch.clamp(levels, min=0.0, max=float(s))

    sign = torch.sign(v)
    dequantized = norm * sign * (levels / float(s))

    total_bits = n * bits_per_param
    size_mb = total_bits / (8.0 * 1024.0 * 1024.0)
    return dequantized, size_mb


def quantize_tensor(tensor: torch.Tensor, num_bits: int = 8):
    """Backward-compatible wrapper around QSGD compression."""
    if num_bits <= 0:
        compressed = _as_flat_float_tensor(tensor)
        return compressed, torch.tensor(0.0), torch.tensor(1.0)

    s = int(2**num_bits)
    compressed, _ = qsgd_compress(tensor, s=s)
    return compressed, torch.tensor(0.0), torch.tensor(1.0)


def dequantize_tensor(q: torch.Tensor, _x_min: torch.Tensor, _scale: torch.Tensor) -> torch.Tensor:
    """Backward-compatible identity dequantization for the QSGD wrapper."""
    return q.detach().clone()


def compressed_size_mb(q: torch.Tensor, num_bits: int) -> float:
    bits = q.numel() * max(num_bits, 1)
    return bits / (8.0 * 1024.0 * 1024.0)
