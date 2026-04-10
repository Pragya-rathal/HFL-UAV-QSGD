### compression.py
import torch
import math


# ─────────────────────────────────────────────
# Top-K + Error Feedback
# ─────────────────────────────────────────────

def topk_compress(update, residual, ratio):
    """
    Apply Top-K compression with error feedback.
    Returns: compressed_update (sparse tensor), new_residual, k
    """
    u = update + residual
    n = len(u)
    k = max(1, int(ratio * n))

    # Keep top-k magnitude entries
    abs_u = u.abs()
    _, top_indices = torch.topk(abs_u, k)

    compressed = torch.zeros_like(u)
    compressed[top_indices] = u[top_indices]

    new_residual = u - compressed
    return compressed, new_residual, k


def topk_message_size_MB(k, full_precision=True):
    """
    Top-K message size:
    values = k * 32 bits, indices = k * 32 bits
    """
    bits = k * 32 + k * 32
    return bits / (8 * 1024 * 1024)


# ─────────────────────────────────────────────
# QSGD Quantization
# ─────────────────────────────────────────────

def qsgd_quantize(update, s):
    """
    QSGD stochastic quantization to s levels.
    Returns: quantized values, signs, norm (for dequantization), s
    """
    norm = update.norm()
    if norm == 0:
        return torch.zeros_like(update), torch.ones_like(update), norm, s

    normalized = update / norm
    signs = torch.sign(normalized)
    signs[signs == 0] = 1.0
    abs_norm = normalized.abs()

    # Stochastic rounding
    floor_vals = torch.floor(abs_norm * s)
    prob = abs_norm * s - floor_vals
    rand = torch.rand_like(prob)
    quantized = floor_vals + (rand < prob).float()
    quantized = torch.clamp(quantized, 0, s)

    return quantized, signs, norm, s


def qsgd_dequantize(quantized, signs, norm, s):
    """Dequantize QSGD-compressed update."""
    return (signs * quantized / s) * norm


def qsgd_message_size_MB(n, s):
    """
    QSGD message size: each parameter = log2(s) bits
    """
    bits_per_param = math.log2(max(2, s))
    bits = n * bits_per_param
    return bits / (8 * 1024 * 1024)


def full_precision_size_MB(n):
    """Full precision: 32 bits per parameter."""
    return (n * 32) / (8 * 1024 * 1024)


# ─────────────────────────────────────────────
# Quorum Selection
# ─────────────────────────────────────────────

def select_quorum(devices, device_ids, fraction, K, current_round, rng=None):
    """
    Select top-fraction devices by quorum score.
    Enforces rotation: each device must participate at least once every K rounds.
    Returns selected device_ids.
    """
    import numpy as np
    if rng is None:
        rng = np.random.RandomState(current_round)

    n = len(device_ids)
    num_select = max(1, int(fraction * n))

    scores = {}
    for did in device_ids:
        scores[did] = devices[did].quorum_score(rng=rng)

    # Check rotation: devices that haven't participated in K rounds
    forced = []
    for did in device_ids:
        last = devices[did].last_participated
        if last == -1 or (current_round - last) >= K:
            forced.append(did)

    # Sort by score
    sorted_ids = sorted(device_ids, key=lambda d: scores[d], reverse=True)
    selected = sorted_ids[:num_select]

    # Add forced rotation devices
    selected_set = set(selected)
    for did in forced:
        selected_set.add(did)

    # If somehow empty, fallback
    if len(selected_set) == 0:
        selected_set = {device_ids[0]}

    return list(selected_set)
