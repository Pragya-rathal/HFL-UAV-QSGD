from typing import List, Tuple


def topk_with_error_feedback(update: List[float], residual: float, ratio: float = 0.1) -> Tuple[List[float], float]:
    combined = [v + residual / max(1, len(update)) for v in update]
    k = max(1, int(len(combined) * ratio))
    ranked = sorted(range(len(combined)), key=lambda i: abs(combined[i]), reverse=True)
    keep = set(ranked[:k])
    compressed = [combined[i] if i in keep else 0.0 for i in range(len(combined))]
    new_residual = sum(combined[i] for i in range(len(combined)) if i not in keep)
    return compressed, new_residual


def qsgd_quantize(update: List[float], levels: int = 8) -> List[float]:
    if not update:
        return []
    max_abs = max(abs(v) for v in update) or 1.0
    step = max_abs / levels
    quantized = []
    for v in update:
        q = round(v / step) * step
        quantized.append(q)
    return quantized


def estimate_bits(update: List[float], mode: str = "full") -> int:
    n = len(update)
    if mode == "topk":
        non_zero = sum(1 for v in update if v != 0.0)
        return non_zero * 64
    if mode == "qsgd":
        return n * 8
    return n * 32
