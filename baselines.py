"""
baselines.py — Modern FL baselines integrated into the HFL pipeline.

Implemented
───────────
FedProx (Li et al., MLSys 2020). Adds μ/2 ‖θ − θ_global‖² to the local
objective. Cleanly compatible with HFL — the proximal term is purely
per-client and survives any aggregator above the client tier.

Deliberately NOT implemented — see SCAFFOLD_NOTE
────────────────────────────────────────────────
SCAFFOLD (Karimireddy et al., ICML 2020). Convergence proof assumes a flat
client→server topology; pushing control variates through a cluster-head
layer requires a non-trivial derivation not present in the literature.
We document this as the published reason for the omission.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from model import clone_model, get_flat_params, set_flat_params, count_parameters
from compression import full_update_size_mb


# ─────────────────────────────────────────────────────────────────────────────
# FedProx local training
# ─────────────────────────────────────────────────────────────────────────────

def fedprox_local_train(model, loader, global_flat, epochs, lr,
                        momentum, weight_decay, mu):
    """
    SGD + momentum + weight decay, with the FedProx proximal regulariser:
        L_i(θ) = CE(θ; D_i) + (μ/2) ‖θ − θ_global‖²

    Reported loss is the CE component only (so it's comparable across methods).
    """
    model.train()
    optim = torch.optim.SGD(model.parameters(), lr=lr,
                            momentum=momentum, weight_decay=weight_decay)
    total_ce, total_n = 0.0, 0
    for _ in range(epochs):
        for X, y in loader:
            optim.zero_grad(set_to_none=True)
            logits = model(X)
            ce = F.cross_entropy(logits, y)
            local_flat = torch.cat([p.reshape(-1) for p in model.parameters()])
            prox = 0.5 * mu * (local_flat - global_flat).pow(2).sum()
            (ce + prox).backward()
            optim.step()
            n = y.numel()
            total_ce += float(ce.item()) * n
            total_n  += n
    return model, total_ce / max(total_n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Method wrapper (matches the signature federated.run_method dispatches to)
# ─────────────────────────────────────────────────────────────────────────────

def run_fedprox_method(global_model_init, train_loaders, test_loader,
                       devices, clusters, head_ids, cfg, mu: float = 0.01) -> List[Dict]:
    """
    FedProx under HFL.
      • intra-cluster: FedProx local training → FedAvg
      • inter-cluster (heads → UAV): FedAvg
      • all devices participate (matches the canonical paper)
      • full-precision communication (compression is orthogonal; we keep
        FedProx clean to make the comparison interpretable)
    """
    # reuse helpers from the existing federated.py
    from federated import fedavg, evaluate, compute_round_latency

    n_params = count_parameters(global_model_init)
    full_size = full_update_size_mb(n_params, cfg.model_bits)

    global_model = clone_model(global_model_init)
    history: List[Dict] = []

    for rnd in range(cfg.num_rounds):
        global_flat = get_flat_params(global_model).clone()

        cluster_flats: List[torch.Tensor] = []
        cluster_comm_mb = 0.0
        round_losses: List[float] = []

        for head_id, member_ids in clusters.items():
            local_flats = []
            for dev_id in member_ids:
                local_model = clone_model(global_model)
                local_model, loss_i = fedprox_local_train(
                    local_model, train_loaders[dev_id], global_flat,
                    cfg.local_epochs, cfg.lr, cfg.momentum, cfg.weight_decay, mu,
                )
                round_losses.append(loss_i)
                local_flats.append(get_flat_params(local_model))
                cluster_comm_mb += full_size
            if local_flats:
                cluster_flats.append(fedavg(global_flat, local_flats))

        cluster_uav_mb = len(cluster_flats) * full_size
        if cluster_flats:
            set_flat_params(global_model, fedavg(global_flat, cluster_flats))

        # Latency (same model as the other methods for an apples-to-apples plot)
        active_per_cluster = clusters
        t_round, t_mean, t_p75, ctl = compute_round_latency(
            active_per_cluster, devices, full_size,
            cfg.base_compute_time, cfg.agg_head_time, cfg.uav_comm_base, head_ids,
        )

        acc, eval_loss = evaluate(global_model, test_loader)
        history.append({
            "round": rnd + 1,
            "accuracy": float(acc),
            "loss": float(np.mean(round_losses)) if round_losses else 0.0,
            "eval_loss": float(eval_loss),
            "latency_round": float(t_round),
            "latency_mean":  float(t_mean),
            "latency_p75":   float(t_p75),
            "comm_device_to_head_mb": float(cluster_comm_mb),
            "comm_head_to_uav_mb":    float(cluster_uav_mb),
            "comm_total_mb":          float(cluster_comm_mb + cluster_uav_mb),
            "active_devices":         sum(len(v) for v in clusters.values()),
            "cluster_times":          ctl,
        })
        print(f"  [fedprox]       R{rnd+1:3d}/{cfg.num_rounds} "
              f"Acc={acc:.4f} Loss={eval_loss:.4f} "
              f"Lat={t_round:.3f}s Comm={cluster_comm_mb+cluster_uav_mb:.2f}MB",
              flush=True)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# HierFAVG  (Liu et al., ICC 2020)
# ─────────────────────────────────────────────────────────────────────────────

def run_hierfavg_method(global_model_init, train_loaders, test_loader,
                        devices, clusters, head_ids, cfg) -> List[Dict]:
    """
    HierFAVG: two-level FedAvg.

    Each global round contains  κ_2 = cfg.edge_rounds  intra-cluster
    aggregations. Between successive intra-cluster aggregations, the edge
    (cluster-head) model is broadcast back to its members, who then continue
    training from that aggregated point. After all κ_2 edge rounds, cluster-
    head models are averaged at the UAV.

    This is the canonical hierarchical-FL aggregator and the most direct
    prior-art comparator for our framework. With κ_2 = 1 the algorithm
    degenerates to our `clustered_fl` baseline; the original paper uses 3.

    Compression: none (matches the canonical paper). Participation: full.
    """
    from federated import evaluate, fedavg, local_train as _local_train

    n_params = count_parameters(global_model_init)
    full_size = full_update_size_mb(n_params, cfg.model_bits)
    edge_rounds = max(1, int(getattr(cfg, "edge_rounds", 3)))

    global_model = clone_model(global_model_init)
    history: List[Dict] = []

    for rnd in range(cfg.num_rounds):
        global_flat = get_flat_params(global_model).clone()

        # κ_2 intra-cluster aggregation cycles ──────────────────────────────
        edge_flats = {h: global_flat.clone() for h in clusters}
        round_losses: List[float] = []
        cluster_comm_mb = 0.0      # device → head traffic, summed over κ_2 rounds

        for _ in range(edge_rounds):
            new_edge_flats = {}
            for head_id, member_ids in clusters.items():
                edge_flat = edge_flats[head_id]
                local_flats = []
                for dev_id in member_ids:
                    local_model = clone_model(global_model)
                    set_flat_params(local_model, edge_flat)   # start from edge model
                    local_model, loss_i = _local_train(
                        local_model, train_loaders[dev_id],
                        cfg.local_epochs, cfg.lr, cfg.momentum, cfg.weight_decay,
                    )
                    round_losses.append(loss_i)
                    local_flats.append(get_flat_params(local_model))
                    cluster_comm_mb += full_size               # one device → head upload
                if local_flats:
                    new_edge_flats[head_id] = fedavg(edge_flat, local_flats)
                else:
                    new_edge_flats[head_id] = edge_flat
            edge_flats = new_edge_flats

        # Global aggregation (cluster heads → UAV), once per global round ───
        cluster_uav_mb = len(edge_flats) * full_size
        new_global_flat = fedavg(global_flat, list(edge_flats.values()))
        set_flat_params(global_model, new_global_flat)

        # Latency: κ_2 × (intra-cluster compute + intra-cluster comm) + UAV gather
        intra_times = []
        all_indiv = []
        for _, mids in clusters.items():
            dt = [devices[i].total_time(cfg.base_compute_time, full_size) for i in mids]
            all_indiv.extend(dt)
            intra_times.append(max(dt) + cfg.agg_head_time)
        t_round = edge_rounds * max(intra_times) + cfg.uav_comm_base * len(head_ids)
        t_mean = float(np.mean(all_indiv))
        t_p75  = float(np.percentile(all_indiv, 75))

        acc, eval_loss = evaluate(global_model, test_loader)
        history.append({
            "round": rnd + 1,
            "accuracy": float(acc),
            "loss": float(np.mean(round_losses)) if round_losses else 0.0,
            "eval_loss": float(eval_loss),
            "latency_round": float(t_round),
            "latency_mean":  float(t_mean),
            "latency_p75":   float(t_p75),
            "comm_device_to_head_mb": float(cluster_comm_mb),
            "comm_head_to_uav_mb":    float(cluster_uav_mb),
            "comm_total_mb":          float(cluster_comm_mb + cluster_uav_mb),
            "active_devices":         sum(len(v) for v in clusters.values()),
            "cluster_times":          [edge_rounds * t for t in intra_times],
            "edge_rounds":            edge_rounds,
        })
        print(f"  [hierfavg]      R{rnd+1:3d}/{cfg.num_rounds} "
              f"Acc={acc:.4f} Loss={eval_loss:.4f} "
              f"Lat={t_round:.3f}s Comm={cluster_comm_mb+cluster_uav_mb:.2f}MB "
              f"κ₂={edge_rounds}", flush=True)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# SCAFFOLD note (for the paper appendix / discussion)
# ─────────────────────────────────────────────────────────────────────────────

SCAFFOLD_NOTE = """\
Why SCAFFOLD is omitted from the experimental comparison
========================================================
SCAFFOLD (Karimireddy et al., ICML 2020) reduces client drift in federated
optimisation by maintaining per-client control variates c_i and a server
control variate c, correcting local SGD steps by  g − c_i + c. Its
convergence proof assumes a flat one-tier topology in which the server
aggregates both model parameters and control variates within a single
FedAvg-like step.

In our hierarchical FL setting (clients → cluster heads → UAV) the mapping
is ambiguous in two ways:

(i)  Intra-cluster SCAFFOLD only. Control variates are exchanged between
     each cluster head and its members; cluster-head models are then
     FedAvg-aggregated at the UAV. This is a valid 1-tier SCAFFOLD inside
     each cluster, but the inter-tier drift between heads and the global
     model remains uncorrected. The convergence guarantee from the
     original paper does not transfer.

(ii) Two-level control variates. A natural extension would maintain
     control variates at both tiers (c_i for clients, c_h for cluster
     heads, c for the UAV) and apply two corrections per round. We are
     not aware of a derivation of this scheme in the literature; producing
     and validating one is itself a research contribution.

We therefore compare against FedProx, whose proximal regulariser is purely
per-client and composes trivially under any aggregator above the client
tier. This choice is deliberate and documented; future work could derive
a hierarchical analogue of SCAFFOLD and revisit the comparison.
"""
