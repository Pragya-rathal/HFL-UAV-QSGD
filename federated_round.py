"""
federated_round.py — The per-round primal-dual loop.

This file is the focal contribution of the paper. It implements:

  ┌─ each communication round t ───────────────────────────────────────────┐
  │                                                                        │
  │   1.  EVOLVE        G_t  ← evolve(G_{t-1}, bw_noise, dropout, mobility)│
  │   2.  METRICS       cheap every round; APL/betweenness every K_topo    │
  │   3.  TOPOLOGY UTIL T_i  = closeness centrality within cluster subgraph│
  │   4.  PRIMAL POLICY q_i  = σ(λ_D − λ_C)                                │
  │                     S_i  = γT_i − (1+λ_C)ĉ_i − (1+λ_L)l̂_i − (1+λ_D)d̂_i │
  │                     pick top-N_k per cluster by S_i                    │
  │   5.  TRAIN         local SGD on selected devices                      │
  │   6.  AGGREGATE     intra-cluster FedAvg → cluster-head → UAV          │
  │   7.  OBSERVE       C_t, L_t, D_t                                      │
  │   8.  DUAL UPDATE   λ ← [λ + η(g_t − targets)]_+                       │
  │   9.  RE-ELECT      heads every cfg.rehead_every rounds                │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

Wiring into the existing `federated.py`:
  • Method 'topoco' calls run_topoco_method() inside its per-round loop.
  • Existing methods (standard_fl, clustered_fl, topk_ef, qsgd) continue to
    work unchanged → automatic ablation.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

from model import clone_model, get_flat_params, set_flat_params, count_parameters
from compression import (
    topk_compress, topk_decompress, topk_message_size_mb, full_update_size_mb,
)
from devices import IoTDevice
from topology import (
    build_graph, compute_metrics, topology_utility,
    evolve_devices, TopologyEvolutionConfig,
)
from utility import (
    LagrangianState, compute_per_device_score, update_lagrangian,
    update_divergence_proxy, predict_comm_cost_mb, predict_latency_s,
)
from schedules import (
    ScheduleConfig, adaptive_compression, adaptive_participation,
    schedule_diagnostics,
)
from clustering import reelect_heads


# ─────────────────────────────────────────────────────────────────────────────
# Local training (self-contained copy — avoids circular import with federated.py)
# ─────────────────────────────────────────────────────────────────────────────

def _local_train(model, loader, epochs, lr, momentum, weight_decay):
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=lr,
                            momentum=momentum, weight_decay=weight_decay)
    total_loss, total_n = 0.0, 0
    for _ in range(epochs):
        for X, y in loader:
            optim.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X), y)
            loss.backward(); optim.step()
            n = y.numel()
            total_loss += float(loss.item()) * n
            total_n += n
    return model, total_loss / max(total_n, 1)


@torch.no_grad()
def _evaluate(model, loader):
    model.eval()
    correct = total = 0; tot_loss = 0.0
    for X, y in loader:
        logits = model(X)
        tot_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
        correct  += int((logits.argmax(1) == y).sum().item())
        total    += int(y.numel())
    model.train()
    return correct / total, tot_loss / total


# ─────────────────────────────────────────────────────────────────────────────
# One round of TopoCo
# ─────────────────────────────────────────────────────────────────────────────

def run_topoco_round(
    *, round_idx: int,
    global_model,
    train_loaders,
    test_loader,
    devices: List[IoTDevice],
    head_ids: List[int],
    clusters: Dict[int, List[int]],
    state: LagrangianState,
    sched_cfg: ScheduleConfig,
    cfg,
    rng: np.random.RandomState,
    divergence_proxy: Dict[int, float],
):
    """
    Executes one round of the primal-dual loop. Returns:
      (history_row, new_head_ids, new_clusters, new_divergence_proxy)
    """

    # ── 1. Evolve devices + rebuild graph ──────────────────────────────────
    evo = TopologyEvolutionConfig(
        bw_noise_std=cfg.bw_noise_std,
        dropout_prob=cfg.dropout_prob,
        mobility_step=cfg.mobility_step,
        area_size=cfg.area_size,
    )
    evolve_devices(devices, evo, rng)
    G = build_graph(devices, cfg.r_comm)

    # ── 2. Compute topology metrics (cheap; expensive every K_topo rounds) ─
    compute_expensive = ((round_idx + 1) % cfg.K_topo == 0)
    metrics = compute_metrics(G, round_idx, compute_expensive=compute_expensive)

    # ── 3. Topology utility T_i (cluster-aware closeness) ──────────────────
    T = topology_utility(G, clusters)

    # ── 4. Primal policy: compression q and per-device score S_i ───────────
    if cfg.adaptive_compression_enabled:
        q_assign = adaptive_compression(devices, state, sched_cfg)
    else:
        q_assign = {d.device_id: cfg.topk_fraction for d in devices}

    n_params = count_parameters(global_model)
    scores = compute_per_device_score(
        devices, T, state, n_params,
        cfg.base_compute_time, q_assign, divergence_proxy,
    )

    # ── 4b. Participation: top-N per cluster by S_i ─────────────────────────
    if cfg.adaptive_participation_enabled:
        active_per_cluster = adaptive_participation(clusters, scores, state, sched_cfg)
    else:
        dev_map = {d.device_id: d for d in devices}
        active_per_cluster = {
            h: [m for m in members if dev_map[m].is_active][
                : max(sched_cfg.floor_m,
                      int(cfg.rho_max * len(members)))
            ]
            for h, members in clusters.items()
        }

    total_active = sum(len(v) for v in active_per_cluster.values())

    # ── 5–6. Train, compress, aggregate ─────────────────────────────────────
    global_flat = get_flat_params(global_model).clone()
    full_size_mb = full_update_size_mb(n_params, cfg.model_bits)

    cluster_flats: List[torch.Tensor] = []
    cluster_comm_mb = 0.0
    round_losses: List[float] = []
    update_norms: Dict[int, float] = {}      # for next round's divergence proxy

    for head_id, member_ids in active_per_cluster.items():
        local_flats: List[torch.Tensor] = []
        for dev_id in member_ids:
            local_model = clone_model(global_model)
            local_model, loss_i = _local_train(
                local_model, train_loaders[dev_id],
                cfg.local_epochs, cfg.lr, cfg.momentum, cfg.weight_decay,
            )
            round_losses.append(loss_i)
            raw_update = get_flat_params(local_model) - global_flat

            # Per-device adaptive compression (top-K with q_i)
            q_i = q_assign[dev_id]
            zero_res = torch.zeros_like(raw_update)
            vals, idxs, _ = topk_compress(raw_update, zero_res, q_i)
            delta = topk_decompress(vals, idxs, n_params)
            communicated = global_flat + delta

            cluster_comm_mb += predict_comm_cost_mb(n_params, q_i, cfg.model_bits)
            update_norms[dev_id] = float(raw_update.norm().pow(2).item())
            local_flats.append(communicated)

        if local_flats:
            stack = torch.stack(local_flats)
            cluster_flats.append(stack.mean(dim=0))

    cluster_uav_mb = len(cluster_flats) * full_size_mb
    if cluster_flats:
        new_flat = torch.stack(cluster_flats).mean(dim=0)
        set_flat_params(global_model, new_flat)

    # ── 7. Observed signals ─────────────────────────────────────────────────
    # Per-device latency and cluster-level latency
    cluster_lats = []
    all_indiv_lats = []
    for h, mids in active_per_cluster.items():
        if not mids:
            continue
        per_dev = []
        for i in mids:
            q_i = q_assign[i]
            comm_mb = predict_comm_cost_mb(n_params, q_i, cfg.model_bits)
            per_dev.append(predict_latency_s(comm_mb, devices[i].bandwidth,
                                             devices[i].compute_power, cfg.base_compute_time))
        all_indiv_lats.extend(per_dev)
        cluster_lats.append(max(per_dev) + cfg.agg_head_time)

    t_round = (max(cluster_lats) + cfg.uav_comm_base * len(head_ids)) if cluster_lats else 0.0
    t_mean  = float(np.mean(all_indiv_lats)) if all_indiv_lats else 0.0
    t_p75   = float(np.percentile(all_indiv_lats, 75)) if all_indiv_lats else 0.0

    observed_C = cluster_comm_mb + cluster_uav_mb
    observed_L = t_round
    observed_D = float(np.mean(list(update_norms.values()))) if update_norms else 0.0

    # ── 8. Dual update (the principled adaptive mechanism) ─────────────────
    update_lagrangian(state, observed_C, observed_L, observed_D)
    state.snapshot(round_idx, {
        "C_obs": observed_C, "L_obs": observed_L, "D_obs": observed_D,
        "n_selected": total_active, "q_mean": float(np.mean(list(q_assign.values()))),
    })

    # ── 9. Optional head re-election ───────────────────────────────────────
    new_heads, new_clusters = head_ids, clusters
    if cfg.rehead_every > 0 and (round_idx + 1) % cfg.rehead_every == 0:
        new_heads, new_clusters = reelect_heads(devices, G, scores, cfg)

    # ── Eval ───────────────────────────────────────────────────────────────
    acc, eval_loss = _evaluate(global_model, test_loader)

    # ── Divergence proxy for next round ────────────────────────────────────
    new_proxy = update_divergence_proxy(divergence_proxy, update_norms)

    sched_diag = schedule_diagnostics(active_per_cluster, q_assign)

    # History row — schema-compatible with all other methods (superset):
    # base fields present in every method + topoco-specific extensions.
    row = {
        "round": round_idx + 1,
        "accuracy": float(acc),
        "loss": float(np.mean(round_losses)) if round_losses else 0.0,
        "eval_loss": float(eval_loss),
        # Latency — full schema (latency_round, latency_mean, latency_p75)
        "latency_round": float(t_round),
        "latency_mean":  float(t_mean),
        "latency_p75":   float(t_p75),
        # Communication — full schema
        "comm_device_to_head_mb": float(cluster_comm_mb),
        "comm_head_to_uav_mb":    float(cluster_uav_mb),
        "comm_total_mb":          float(observed_C),
        "active_devices":         int(total_active),
        "cluster_times":          cluster_lats,
        # Topology
        "avg_local_cc": metrics.avg_local_cc,
        "apl": metrics.apl if metrics.apl is not None else float("nan"),
        "n_edges": metrics.n_edges,
        "density": metrics.density,
        # Lagrangian state
        "lambda_C": state.lambda_C, "lambda_L": state.lambda_L, "lambda_D": state.lambda_D,
        # Schedule diagnostics
        **{f"sched_{k}": v for k, v in sched_diag.items()},
    }
    return row, new_heads, new_clusters, new_proxy


# ─────────────────────────────────────────────────────────────────────────────
# Method wrapper — loops run_topoco_round for cfg.num_rounds
# This is what federated.run_method dispatches to when method == "topoco".
# ─────────────────────────────────────────────────────────────────────────────

def run_topoco_method(global_model_init, train_loaders, test_loader,
                      devices, clusters, head_ids, cfg):
    """
    Full TopoCo training run. Returns a list[dict] history compatible with
    metrics.history_to_df.

    The rng seed is derived from cfg.seeds: if the caller (run_single) has
    already called set_seed(seed) via torch/numpy, we derive the rng from
    the *first element of cfg.seeds that matches the numpy global state*.
    To guarantee full reproducibility across seeds we accept a seed kwarg
    forwarded through cfg._current_seed (set by run_single before dispatch).
    Falls back to cfg.seeds[0] if not present.
    """
    global_model = clone_model(global_model_init)

    # Use the per-run seed stored by run_single, falling back to seeds[0].
    seed = getattr(cfg, "_current_seed", cfg.seeds[0] if cfg.seeds else 42)

    state = LagrangianState(
        lambda_C=cfg.lambda_C_init, lambda_L=cfg.lambda_L_init, lambda_D=cfg.lambda_D_init,
        eta_C=cfg.eta_C, eta_L=cfg.eta_L, eta_D=cfg.eta_D,
        B_target_mb=cfg.B_target_mb, L_target_s=cfg.L_target_s, D_target=cfg.D_target,
        gamma_T=cfg.gamma_T, lambda_max=cfg.lambda_max,
    )
    sched_cfg = ScheduleConfig(
        q_min=cfg.q_min, q_max=cfg.q_max,
        rho_max=cfg.rho_max, floor_m=cfg.floor_m,
    )
    rng = np.random.RandomState(seed)
    proxy = {d.device_id: 1.0 for d in devices}

    history = []
    cur_heads, cur_clusters = head_ids, clusters

    for rnd in range(cfg.num_rounds):
        row, cur_heads, cur_clusters, proxy = run_topoco_round(
            round_idx=rnd, global_model=global_model,
            train_loaders=train_loaders, test_loader=test_loader,
            devices=devices, head_ids=cur_heads, clusters=cur_clusters,
            state=state, sched_cfg=sched_cfg, cfg=cfg, rng=rng,
            divergence_proxy=proxy,
        )
        history.append(row)
        print(f"  [topoco]        R{row['round']:3d}/{cfg.num_rounds} "
              f"Acc={row['accuracy']:.4f} Loss={row['eval_loss']:.4f} "
              f"Lat={row['latency_round']:.3f}s Comm={row['comm_total_mb']:.2f}MB "
              f"λC={row['lambda_C']:.2f} λL={row['lambda_L']:.2f} λD={row['lambda_D']:.2f} "
              f"q̄={row['sched_q_mean']:.3f}",
              flush=True)

    return history
