"""
Hierarchical Federated Learning engine — GPU (torch) backend.

All 6 methods (standard_fl, clustered_fl, topk_ef, qsgd, topk_quorum, qsgd_quorum)
share identical data / device / cluster assignments per seed.

All training, parameter, residual, compression, and aggregation tensors live on
`cfg.device` (typically CUDA). No host↔device transfer per round.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model import (
    clone_model, get_flat_params, set_flat_params, count_parameters
)
from compression import (
    topk_compress, topk_decompress, topk_message_size_mb,
    qsgd_quantize, qsgd_dequantize, qsgd_message_size_mb,
    full_update_size_mb,
)
from devices import IoTDevice


# ── Local training (torch SGD + momentum) ────────────────────────────────────

def local_train(model, loader, epochs: int, lr: float,
                momentum: float, weight_decay: float):
    """
    SGD with momentum + weight decay. Matches the NumPy version's behaviour:
    momentum buffer is fresh each call (no carry-over across communication rounds).
    """
    model.train()
    optim = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    total_loss = 0.0
    total_n = 0
    for _ in range(epochs):
        for X, y in loader:
            optim.zero_grad(set_to_none=True)
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optim.step()
            n = y.numel()
            total_loss += float(loss.item()) * n
            total_n += n
    return model, total_loss / max(total_n, 1)


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    for X, y in loader:
        logits = model(X)
        # reduction='sum' so total_loss is a true sum over samples
        total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
        correct   += int((logits.argmax(1) == y).sum().item())
        total     += int(y.numel())
    model.train()
    return correct / total, total_loss / total


# ── Quorum selection (CPU-side, scalar logic) ────────────────────────────────

def select_quorum(member_ids: List[int], devices: List[IoTDevice], fraction: float):
    k = max(1, int(np.ceil(fraction * len(member_ids))))
    scored = sorted(
        member_ids,
        key=lambda i: devices[i].compute_power + devices[i].bandwidth,
        reverse=True,
    )
    return scored[:k]


# ── Latency model (scalar, CPU) ──────────────────────────────────────────────

def compute_round_latency(
    active_per_cluster: Dict[int, List[int]],
    devices: List[IoTDevice],
    msg_mb: float,
    base_compute_time: float,
    agg_head_time: float,
    uav_comm_base: float,
    head_ids: List[int],
):
    all_dt, cluster_times = [], []
    for _, mids in active_per_cluster.items():
        dt = [devices[i].total_time(base_compute_time, msg_mb) for i in mids]
        all_dt.extend(dt)
        cluster_times.append(max(dt) + agg_head_time)
    t_round = max(cluster_times) + uav_comm_base * len(head_ids)
    return (
        t_round,
        float(np.mean(all_dt)),
        float(np.percentile(all_dt, 75)),
        cluster_times,
    )


# ── FedAvg (GPU vectorised) ──────────────────────────────────────────────────

def fedavg(global_flat: torch.Tensor,
           local_flats: List[torch.Tensor],
           weights: Optional[List[float]] = None) -> torch.Tensor:
    """Weighted average of flat-param tensors. Single GPU matmul."""
    if not local_flats:
        return global_flat.clone()

    stacked = torch.stack(local_flats)  # (M, D)

    if weights is None:
        w = torch.full((stacked.shape[0],), 1.0 / stacked.shape[0],
                       dtype=stacked.dtype, device=stacked.device)
    else:
        w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device)
        w = w / w.sum()

    return (w.unsqueeze(1) * stacked).sum(dim=0)


# ── Master runner ────────────────────────────────────────────────────────────

def run_method(method, global_model_init, train_loaders, test_loader,
               devices, clusters, head_ids, cfg):
    # ─── New methods (route out before the 6-classic dispatch) ──────────────
    if method == "topoco":
        from federated_round import run_topoco_method
        return run_topoco_method(global_model_init, train_loaders, test_loader,
                                 devices, clusters, head_ids, cfg)
    if method == "fedprox":
        from baselines import run_fedprox_method
        return run_fedprox_method(global_model_init, train_loaders, test_loader,
                                  devices, clusters, head_ids, cfg)
    # ─── End new dispatch — original 6-method code follows unchanged ────────

    n_params  = count_parameters(global_model_init)
    full_size = full_update_size_mb(n_params, cfg.model_bits)

    # Latency-accounting message size (depends on method)
    if method in ("standard_fl", "clustered_fl"):
        latency_msg = full_size
    elif method in ("topk_ef", "topk_quorum"):
        k = max(1, int(n_params * cfg.topk_fraction))
        latency_msg = topk_message_size_mb(k, cfg.model_bits)
    else:
        latency_msg = qsgd_message_size_mb(n_params, cfg.qsgd_levels, cfg.model_bits)

    global_model = clone_model(global_model_init)
    device = next(global_model.parameters()).device

    # Per-device residuals (GPU)
    residuals: Dict[int, torch.Tensor] = {}
    if method in ("topk_ef", "topk_quorum"):
        for dev in devices:
            residuals[dev.device_id] = torch.zeros(n_params, dtype=torch.float32, device=device)

    history: List[Dict] = []

    for rnd in range(cfg.num_rounds):
        global_flat = get_flat_params(global_model).clone()

        # Active set per cluster
        if method in ("topk_quorum", "qsgd_quorum"):
            active_per_cluster = {
                h: select_quorum(mids, devices, cfg.quorum_fraction)
                for h, mids in clusters.items()
            }
        else:
            active_per_cluster = {h: list(mids) for h, mids in clusters.items()}

        if method == "standard_fl":
            active_per_cluster = {0: list(range(len(devices)))}

        total_active = sum(len(v) for v in active_per_cluster.values())

        cluster_flats: List[torch.Tensor] = []
        cluster_comm_mb = 0.0
        round_losses: List[float] = []

        for head_id, member_ids in active_per_cluster.items():
            local_flats: List[torch.Tensor] = []

            for dev_id in member_ids:
                local_model = clone_model(global_model)
                flat_before = global_flat

                local_model, loss = local_train(
                    local_model, train_loaders[dev_id],
                    cfg.local_epochs, cfg.lr,
                    cfg.momentum, cfg.weight_decay,
                )
                round_losses.append(loss)

                raw_update = get_flat_params(local_model) - flat_before

                if method in ("standard_fl", "clustered_fl"):
                    communicated = global_flat + raw_update
                    comm_size    = full_size

                elif method in ("topk_ef", "topk_quorum"):
                    vals, idxs, new_res = topk_compress(
                        raw_update, residuals[dev_id], cfg.topk_fraction
                    )
                    residuals[dev_id] = new_res
                    delta = topk_decompress(vals, idxs, n_params)
                    communicated = global_flat + delta
                    comm_size    = topk_message_size_mb(int(vals.numel()), cfg.model_bits)

                else:  # qsgd / qsgd_quorum
                    q, scale = qsgd_quantize(raw_update, cfg.qsgd_levels)
                    delta    = qsgd_dequantize(q, scale)
                    communicated = global_flat + delta
                    comm_size    = qsgd_message_size_mb(n_params, cfg.qsgd_levels, cfg.model_bits)

                cluster_comm_mb += comm_size
                local_flats.append(communicated)

            if local_flats:
                cluster_flats.append(fedavg(global_flat, local_flats))

        cluster_uav_mb = len(cluster_flats) * full_size

        if cluster_flats:
            new_flat = fedavg(global_flat, cluster_flats)
            set_flat_params(global_model, new_flat)

        # Latency
        if method == "standard_fl":
            dts = [
                devices[i].total_time(cfg.base_compute_time, latency_msg)
                for i in range(len(devices))
            ]
            t_round = max(dts) + cfg.agg_head_time + cfg.uav_comm_base
            t_mean = float(np.mean(dts))
            t_p75  = float(np.percentile(dts, 75))
            cluster_times_list = [t_round]
        else:
            t_round, t_mean, t_p75, cluster_times_list = compute_round_latency(
                active_per_cluster, devices, latency_msg,
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
            "active_devices":         int(total_active),
            "cluster_times":          cluster_times_list,
        })

        print(
            f"  [{method:15s}] R{rnd+1:3d}/{cfg.num_rounds} "
            f"Acc={acc:.4f} Loss={eval_loss:.4f} "
            f"Lat={t_round:.3f}s Comm={cluster_comm_mb+cluster_uav_mb:.2f}MB "
            f"Active={total_active}",
            flush=True,
        )

    return history
