import math
from typing import Dict, Tuple

import torch


def full_precision_size_mb(num_params: int, bits: int = 32) -> float:
    return (num_params * bits) / 8 / (1024**2)


def topk_compress_with_error_feedback(
    delta: Dict[str, torch.Tensor],
    residual: Dict[str, torch.Tensor],
    topk_ratio: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, float]:
    compressed = {}
    new_residual = {}
    total_vals = 0
    sent_vals = 0

    for k, t in delta.items():
        u = t + residual[k]
        flat = u.flatten()
        n = flat.numel()
        k_keep = max(1, int(topk_ratio * n))
        total_vals += n
        sent_vals += k_keep

        _, idx = torch.topk(flat.abs(), k_keep, sorted=False)
        sparse_flat = torch.zeros_like(flat)
        sparse_flat[idx] = flat[idx]
        sparse = sparse_flat.view_as(u)

        compressed[k] = sparse
        new_residual[k] = u - sparse

    # values in fp32 + indices int32
    bits = sent_vals * (32 + 32)
    size_mb = bits / 8 / (1024**2)
    compression_ratio = total_vals / max(sent_vals, 1)
    return compressed, new_residual, size_mb, compression_ratio


def qsgd_quantize(
    delta: Dict[str, torch.Tensor], levels: int
) -> Tuple[Dict[str, torch.Tensor], float, float]:
    quantized = {}
    total_vals = 0

    for k, t in delta.items():
        flat = t.flatten()
        total_vals += flat.numel()
        norm = torch.norm(flat, p=2)
        if norm.item() == 0.0:
            quantized[k] = torch.zeros_like(t)
            continue

        abs_v = flat.abs() * levels / norm
        l = torch.floor(abs_v)
        prob = abs_v - l
        rand = torch.rand_like(prob)
        q = l + (rand < prob).float()
        q = torch.clamp(q, max=levels)
        restored = torch.sign(flat) * norm * q / levels
        quantized[k] = restored.view_as(t)

    bits_per_val = max(1, int(math.ceil(math.log2(levels + 1)))) + 1  # sign + level bits
    bits_overhead = 32  # per tensor norm
    num_tensors = len(delta)
    bits = total_vals * bits_per_val + num_tensors * bits_overhead
    size_mb = bits / 8 / (1024**2)

    full_bits = total_vals * 32
    compression_ratio = full_bits / max(bits, 1)
    return quantized, size_mb, compression_ratio
