"""
utility.py — Per-round selection score & Lagrangian dual updates.

================================================================================
Formal problem
================================================================================
At round t, given graph G_t and per-device features, choose participation set
S_t ⊆ V and per-device compression levels q_i ∈ [q_min, q_max] to solve

  minimize_{S, q}    Σ_{i∈S} c_i(q_i)           (communication cost)
                    + l_max(S, q)               (round-latency)
                    + d̂(S, q)                   (estimated divergence)

  subject to    Σ_{i∈S} c_i(q_i)  ≤ B_t         (bandwidth budget)
                l_max(S, q)        ≤ L_target   (latency target)
                |S ∩ V_k|          ≥ m_k    ∀k  (cluster participation floor)
                q_i ∈ [q_min, q_max]

Lagrangian (with multipliers λ_C, λ_L, λ_D ≥ 0):

  ℒ(S, q, λ) = Σ c_i + l_max + d̂
              + λ_C (Σ c_i − B_t)
              + λ_L (l_max − L_target)
              + λ_D (d̂ − D_target)

The per-device inclusion score is the dual derivative w.r.t. including i:

  S_i  =  γ · T_i    −    (1 + λ_C) · ĉ_i    −    (1 + λ_L) · l̂_i    −    (1 + λ_D) · d̂_i

where T_i is the topology utility from `topology.topology_utility`.

Higher S_i  →  more attractive to include.

================================================================================
Dual updates (subgradient ascent)
================================================================================
At the end of round t, after observing realised C_t, L_t, D_t:

  λ_C ← [λ_C + η_C (C_t − B_t)]_+
  λ_L ← [λ_L + η_L (L_t − L_target)]_+
  λ_D ← [λ_D + η_D (D_t − D_target)]_+

This is the *adaptive weight mechanism*: when a constraint is violated, its
multiplier grows and the next round penalises that dimension more. It is a
standard primal-dual algorithm (Nedić–Ozdaglar 2009, Boyd et al. 2011) — not a
hand-tuned IF/ELSE rule.

================================================================================
Convergence interpretation
================================================================================
Under standard assumptions (bounded primal feasible set, diminishing or
constant-and-small step size η), dual subgradient ascent on a convex relaxation
of this problem converges to a saddle point. Our primal is combinatorial (S is
discrete) so we get an approximate primal–dual gap, but in practice the dual
trajectory still tracks the right trade-off — which is empirically what we
report (multiplier evolution plot).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Cost predictions used inside the score (closed-form, per device i, given q_i)
# ─────────────────────────────────────────────────────────────────────────────

def predict_comm_cost_mb(num_params: int, q_frac: float, bits: int = 32) -> float:
    """
    Communication cost in MB for one device → cluster-head message under Top-K
    compression at fraction q_frac. (Mirrors compression.topk_message_size_mb.)
    """
    k = max(1, int(num_params * q_frac))
    return (k * bits + k * 32) / 8e6


def predict_latency_s(comm_mb: float, bandwidth_mbps: float,
                      compute_power: float, base_compute_time: float) -> float:
    """t_i(q_i) = T_compute_i + T_comm_i(q_i)"""
    t_comp = base_compute_time / max(compute_power, 1e-6)
    t_comm = (comm_mb * 8.0) / max(bandwidth_mbps, 1e-6)
    return float(t_comp + t_comm)


# ─────────────────────────────────────────────────────────────────────────────
# Lagrangian state — the dual variables, persisted across rounds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LagrangianState:
    # Dual variables (non-negative)
    lambda_C: float = 0.5       # bandwidth-cost multiplier
    lambda_L: float = 0.5       # latency multiplier
    lambda_D: float = 0.5       # divergence multiplier

    # Step sizes (constant; can be made diminishing for stricter convergence)
    eta_C: float = 0.10
    eta_L: float = 0.10
    eta_D: float = 0.10

    # Constraint targets (per-round)
    B_target_mb:   float = 50.0   # bandwidth budget per round (sum over devices)
    L_target_s:    float = 5.0    # round-latency target
    D_target:      float = 1.0    # divergence target (in units of avg ‖Δw_i‖²)

    # Fixed coefficient on topology utility (NOT a dual var — it's a prior
    # weight; could also be made dynamic but we keep it fixed for clarity)
    gamma_T: float = 1.0

    # Multiplier caps (avoid runaway in adversarial regimes)
    lambda_max: float = 10.0

    # History (for plots)
    history: List[Dict] = field(default_factory=list)

    def snapshot(self, round_idx: int, observed: Dict) -> None:
        self.history.append({
            "round": round_idx,
            "lambda_C": self.lambda_C, "lambda_L": self.lambda_L, "lambda_D": self.lambda_D,
            **observed,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Per-device score  S_i  (the primal subproblem)
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_device_score(
    devices,
    T: Dict[int, float],
    state: LagrangianState,
    num_params: int,
    base_compute_time: float,
    q_assign: Dict[int, float],
    divergence_proxy: Dict[int, float],
) -> Dict[int, float]:
    """
    S_i  =  γ T_i  −  (1+λ_C) ĉ_i  −  (1+λ_L) l̂_i  −  (1+λ_D) d̂_i

    `q_assign[i]` is the compression fraction this device would use if selected;
    `divergence_proxy[i]` is an estimate of how much skipping device i would hurt
    convergence (e.g. previous-round ‖Δw_i‖² normalised). For devices with no
    history, pass 1.0 (neutral).

    Inactive devices get score −inf so they're never picked.
    """
    S: Dict[int, float] = {}
    # Normalisation: rescale ĉ_i and l̂_i so the score is unit-free
    # (divide by typical scales; here we use the targets themselves)
    c_scale = max(state.B_target_mb / max(len(devices), 1), 1e-6)
    l_scale = max(state.L_target_s, 1e-6)
    d_scale = max(state.D_target, 1e-6)

    for d in devices:
        if not d.is_active:
            S[d.device_id] = -float("inf")
            continue

        q = q_assign.get(d.device_id, 0.1)
        c_i = predict_comm_cost_mb(num_params, q)
        l_i = predict_latency_s(c_i, d.bandwidth, d.compute_power, base_compute_time)
        d_i = divergence_proxy.get(d.device_id, 1.0)

        score = (
            state.gamma_T * T.get(d.device_id, 0.0)
            - (1.0 + state.lambda_C) * (c_i / c_scale)
            - (1.0 + state.lambda_L) * (l_i / l_scale)
            - (1.0 + state.lambda_D) * (d_i / d_scale)
        )
        S[d.device_id] = float(score)
    return S


# ─────────────────────────────────────────────────────────────────────────────
# Dual update  (subgradient ascent on dual variables)
# ─────────────────────────────────────────────────────────────────────────────

def update_lagrangian(
    state: LagrangianState,
    observed_C_mb: float,
    observed_L_s: float,
    observed_D: float,
) -> None:
    """
    λ ← [λ + η · (observed − target)]_+  capped at lambda_max.

    `observed_D` should be the round's average ‖Δw_i‖² across participating
    devices (or any consistent divergence signal; what matters is consistency
    across rounds).
    """
    state.lambda_C = float(np.clip(
        state.lambda_C + state.eta_C * (observed_C_mb - state.B_target_mb),
        0.0, state.lambda_max,
    ))
    state.lambda_L = float(np.clip(
        state.lambda_L + state.eta_L * (observed_L_s - state.L_target_s),
        0.0, state.lambda_max,
    ))
    state.lambda_D = float(np.clip(
        state.lambda_D + state.eta_D * (observed_D - state.D_target),
        0.0, state.lambda_max,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Divergence proxy
# ─────────────────────────────────────────────────────────────────────────────

def update_divergence_proxy(
    proxy: Dict[int, float],
    last_updates_norm: Dict[int, float],
    ema_alpha: float = 0.5,
) -> Dict[int, float]:
    """
    EMA over per-device update norms.  Devices that contribute large updates
    are 'higher divergence cost to skip' — so they get a higher d̂_i and the
    -λ_D d̂_i term penalises skipping them.

    Important: by construction we *want* to keep high-divergence devices
    participating.  Make sure the sign convention is right where this is used
    (we subtract λ_D · d̂_i in the score, but in `compute_per_device_score`
    we want higher d̂_i to *reduce* the score, which would push us *away* from
    keeping them — wrong direction!)

    The fix is to interpret d̂_i as the *cost of NOT selecting i*. In the
    selection objective, the term that captures "I should keep this device"
    is +λ_D d̂_i  (reward for inclusion when divergence pressure is high).
    See the per-device score formula at the top — we *add* not subtract.

    Implementation note: in compute_per_device_score above we currently
    subtract.  We resolve this by passing divergence_proxy[i] as a *negative*
    value when it represents 'expensive to skip', so the −(1+λ_D)·d̂_i term
    becomes positive.  Cleaner: rewrite the score to *add* λ_D·d̂_i.
    For now we use the convention: d̂_i = max_norm − ‖Δw_i‖²  (so important
    devices have small d̂_i, score penalised less).
    """
    out = dict(proxy)
    if not last_updates_norm:
        return out
    max_norm = max(last_updates_norm.values()) + 1e-9
    for nid, n in last_updates_norm.items():
        # 'd̂_i' = how cheap is it to skip this device (low if device is important)
        cost_to_skip = 1.0 - (n / max_norm)        # 0 = important, 1 = unimportant
        out[nid] = ema_alpha * out.get(nid, 1.0) + (1 - ema_alpha) * cost_to_skip
    return out
