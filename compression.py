from typing import Tuple

import torch



def topk_with_error_feedback(update: torch.Tensor, residual: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor, int]:
    u = update + residual
    k = max(1, int(ratio * u.numel()))
    _, idx = torch.topk(torch.abs(u), k)
    compressed = torch.zeros_like(u)
    compressed[idx] = u[idx]
    new_residual = u - compressed
    bits = k * 32 + k * 32
    return compressed, new_residual, bits


def qsgd_quantize(update: torch.Tensor, levels: int) -> Tuple[torch.Tensor, int]:
    if torch.all(update == 0):
        return update.clone(), 32
    norm = torch.norm(update, p=2)
    scaled = torch.abs(update) * levels / (norm + 1e-12)
    lower = torch.floor(scaled)
    prob = scaled - lower
    rand = torch.rand_like(prob)
    q = lower + (rand < prob).float()
    q = torch.clamp(q, max=levels)
    quant = torch.sign(update) * norm * q / levels
    bits_per_val = int(torch.ceil(torch.log2(torch.tensor(levels + 1))).item()) + 1
    bits = update.numel() * bits_per_val + 32
    return quant, bits


def full_precision_bits(num_params: int) -> int:
    return num_params * 32
