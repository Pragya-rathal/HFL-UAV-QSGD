import torch


def quantize_tensor(tensor: torch.Tensor, num_bits: int = 8):
    if num_bits <= 0:
        return tensor.clone(), torch.tensor(0.0), torch.tensor(1.0)

    x = tensor.detach()
    x_min = x.min()
    x_max = x.max()
    denom = (2**num_bits) - 1
    scale = (x_max - x_min) / max(denom, 1)

    if torch.isclose(scale, torch.tensor(0.0, dtype=scale.dtype)):
        q = torch.zeros_like(x, dtype=torch.int32)
        return q, x_min, torch.tensor(0.0, dtype=x.dtype)

    q = torch.clamp(torch.round((x - x_min) / scale), 0, denom).to(torch.int32)
    return q, x_min, scale


def dequantize_tensor(q: torch.Tensor, x_min: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if torch.isclose(scale, torch.tensor(0.0, dtype=scale.dtype)):
        return torch.zeros_like(q, dtype=torch.float32) + x_min
    return q.to(torch.float32) * scale + x_min


def compressed_size_mb(q: torch.Tensor, num_bits: int) -> float:
    bits = q.numel() * max(num_bits, 1)
    return bits / (8.0 * 1024.0 * 1024.0)
