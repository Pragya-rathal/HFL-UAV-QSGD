"""
schedules.py — Adaptive compression and participation.

These are the *primal* policy functions: given the current Lagrangian state
λ = (λ_C, λ_L, λ_D), choose per-device compression q_i and the participation
subset S_t ⊆ V.

================================================================================
Adaptive compression q_i  ∈ [q_min, q_max]
================================================================================
Sigmoid interpolation driven by (λ_D − λ_C):

  q_i  =  q_min  +  (q_max − q_min) · σ(λ_D − λ_C − β)

  σ(z) = 1 / (1 + exp(−z))

Interpretation: when divergence pressure exceeds bandwidth pressure
(λ_D > λ_C), keep more gradient information (higher q, lighter compression).
When bandwidth pressure dominates, compress harder.

β is a small bias (default 0) so q sits near (q_min+q_max)/2 at equilibrium.

================================================================================
Adaptive participation
================================================================================
Per cluster k, pick the top-N_k devices by score S_i, with floor m_k.

  N_k = max(m_k, ⌊ρ_k · |V_k| · σ(λ_D − λ_L − β')⌋)

So when divergence pressure dominates → more participants.
When latency pressure dominates → fewer (faster) participants.

The cluster floor m_k guarantees we don't degenerate to empty clusters.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math


@dataclass
class ScheduleConfig:
    # Compression bounds
    q_min: float = 0.02         # most aggressive sparsity (2 % kept)
    q_max: float = 0.50         # lightest (50 % kept)
    q_bias: float = 0.0

    # Participation bounds
    rho_max: float = 1.0        # fraction of cluster that *can* participate
    floor_m: int = 2            # minimum participants per cluster
    p_bias: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive compression
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z); return 1.0 / (1.0 + e)
    e = math.exp(z); return e / (1.0 + e)


def adaptive_compression(
    devices,
    state,          # LagrangianState (forward-declared)
    cfg: ScheduleConfig,
) -> Dict[int, float]:
    """
    Returns {device_id → q_i}. Pure function of λ_D − λ_C — uniform across
    devices in this version. (Per-device variants are a clean extension:
    devices with low bandwidth could be pushed further toward q_min.)
    """
    z = state.lambda_D - state.lambda_C - cfg.q_bias
    q = cfg.q_min + (cfg.q_max - cfg.q_min) * _sigmoid(z)
    return {d.device_id: q for d in devices}


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive participation
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_participation(
    clusters: Dict[int, List[int]],
    scores:   Dict[int, float],
    state,
    cfg: ScheduleConfig,
) -> Dict[int, List[int]]:
    """
    For each cluster, return the top-N_k device IDs by S_i, with N_k tuned by
    (λ_D − λ_L).  Floors at cfg.floor_m, ceilings at cfg.rho_max·|V_k|.
    """
    z = state.lambda_D - state.lambda_L - cfg.p_bias
    rho = cfg.rho_max * _sigmoid(z)

    out: Dict[int, List[int]] = {}
    for head_id, members in clusters.items():
        active_members = [m for m in members if scores.get(m, float("-inf")) > float("-inf")]
        if not active_members:
            out[head_id] = []
            continue

        N_k = max(cfg.floor_m, int(math.floor(rho * len(active_members))))
        N_k = min(N_k, len(active_members))

        ranked = sorted(active_members, key=lambda i: scores[i], reverse=True)
        out[head_id] = ranked[:N_k]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def schedule_diagnostics(
    selected: Dict[int, List[int]],
    q_assign: Dict[int, float],
) -> Dict:
    """Returns small dict used by the plotting / metrics layer."""
    n_sel = sum(len(v) for v in selected.values())
    if not q_assign:
        return {"n_selected": n_sel, "q_mean": 0.0, "q_min": 0.0, "q_max": 0.0}
    qs = list(q_assign.values())
    return {
        "n_selected": n_sel,
        "q_mean": float(sum(qs) / len(qs)),
        "q_min":  float(min(qs)),
        "q_max":  float(max(qs)),
    }
