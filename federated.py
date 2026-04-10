"""
Hierarchical Federated Learning engine – NumPy backend.
All 6 methods share identical data / device / cluster assignments per seed.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from model import MLP, clone_model, get_flat_params, set_flat_params, count_parameters, _softmax, _xe
from compression import (
    topk_compress, topk_decompress, topk_message_size_mb,
    qsgd_quantize, qsgd_dequantize, qsgd_message_size_mb,
    full_update_size_mb,
)
from devices import IoTDevice


# ── Local training ────────────────────────────────────────────────────────────

def local_train(model, loader, epochs, lr, momentum, weight_decay):
    vel = None
    total_loss, total_n = 0.0, 0
    for _ in range(epochs):
        for X, y in loader:
            logits = model.forward(X)
            grads, loss = model.backward(logits, y)
            vel = model.apply_grads(grads, lr, momentum, weight_decay, vel)
            total_loss += loss * len(y); total_n += len(y)
    return model, total_loss / max(total_n, 1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    correct, total, total_loss = 0, 0, 0.0
    for X, y in loader:
        logits = model.forward(X)
        from model import _softmax, _xe
        p = _softmax(logits)
        total_loss += _xe(p, y) * len(y)
        correct += (logits.argmax(1) == y).sum()
        total   += len(y)
    return correct / total, total_loss / total

# expose helpers used in evaluate
from model import _softmax, _xe


# ── Quorum ────────────────────────────────────────────────────────────────────

def select_quorum(member_ids, devices, fraction):
    k = max(1, int(np.ceil(fraction * len(member_ids))))
    scored = sorted(member_ids,
                    key=lambda i: devices[i].compute_power + devices[i].bandwidth,
                    reverse=True)
    return scored[:k]


# ── Latency model ─────────────────────────────────────────────────────────────

def compute_round_latency(active_per_cluster, devices, msg_mb,
                          base_compute_time, agg_head_time, uav_comm_base, head_ids):
    all_dt, cluster_times = [], []
    for head_id, mids in active_per_cluster.items():
        dt = [devices[i].total_time(base_compute_time, msg_mb) for i in mids]
        all_dt.extend(dt)
        cluster_times.append(max(dt) + agg_head_time)
    t_round = max(cluster_times) + uav_comm_base * len(head_ids)
    return (t_round,
            float(np.mean(all_dt)),
            float(np.percentile(all_dt, 75)),
            cluster_times)


# ── FedAvg ────────────────────────────────────────────────────────────────────

def fedavg(global_flat, local_flats, weights=None):
    if not local_flats: return global_flat.copy()
    if weights is None: weights = [1.0 / len(local_flats)] * len(local_flats)
    tw = sum(weights)
    agg = np.zeros_like(global_flat)
    for f, w in zip(local_flats, weights):
        agg += (w / tw) * f
    return agg


# ── Master runner ─────────────────────────────────────────────────────────────

def run_method(method, global_model_init, train_loaders, test_loader,
               devices, clusters, head_ids, cfg):
    n_params  = count_parameters(global_model_init)
    full_size = full_update_size_mb(n_params, cfg.model_bits)

    if method in ("standard_fl", "clustered_fl"):
        latency_msg = full_size
    elif method in ("topk_ef", "topk_quorum"):
        k = max(1, int(n_params * cfg.topk_fraction))
        latency_msg = topk_message_size_mb(k, cfg.model_bits)
    else:
        latency_msg = qsgd_message_size_mb(n_params, cfg.qsgd_levels, cfg.model_bits)

    global_model = clone_model(global_model_init)
    residuals = {}
    if method in ("topk_ef", "topk_quorum"):
        for dev in devices:
            residuals[dev.device_id] = np.zeros(n_params, np.float32)

    history = []

    for rnd in range(cfg.num_rounds):
        global_flat = get_flat_params(global_model).copy()

        # Active devices per cluster
        if method in ("topk_quorum", "qsgd_quorum"):
            active_per_cluster = {h: select_quorum(mids, devices, cfg.quorum_fraction)
                                  for h, mids in clusters.items()}
        else:
            active_per_cluster = {h: list(mids) for h, mids in clusters.items()}

        if method == "standard_fl":
            active_per_cluster = {0: list(range(len(devices)))}

        total_active = sum(len(v) for v in active_per_cluster.values())

        cluster_flats, cluster_comm_mb, round_losses = [], 0.0, []

        for head_id, member_ids in active_per_cluster.items():
            local_flats = []
            for dev_id in member_ids:
                local_model = clone_model(global_model)
                flat_before = global_flat

                local_model, loss = local_train(local_model, train_loaders[dev_id],
                                                cfg.local_epochs, cfg.lr,
                                                cfg.momentum, cfg.weight_decay)
                round_losses.append(loss)
                raw_update = get_flat_params(local_model) - flat_before

                if method in ("standard_fl", "clustered_fl"):
                    communicated = global_flat + raw_update
                    comm_size    = full_size

                elif method in ("topk_ef", "topk_quorum"):
                    vals, idxs, new_res = topk_compress(raw_update, residuals[dev_id],
                                                        cfg.topk_fraction)
                    residuals[dev_id] = new_res
                    delta = topk_decompress(vals, idxs, n_params)
                    communicated = global_flat + delta
                    comm_size    = topk_message_size_mb(vals.size, cfg.model_bits)

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
            dts = [devices[i].total_time(cfg.base_compute_time, latency_msg)
                   for i in range(len(devices))]
            t_round = max(dts) + cfg.agg_head_time + cfg.uav_comm_base
            t_mean, t_p75 = float(np.mean(dts)), float(np.percentile(dts, 75))
            cluster_times_list = [t_round]
        else:
            t_round, t_mean, t_p75, cluster_times_list = compute_round_latency(
                active_per_cluster, devices, latency_msg,
                cfg.base_compute_time, cfg.agg_head_time, cfg.uav_comm_base, head_ids)

        acc, eval_loss = evaluate(global_model, test_loader)

        history.append({
            "round": rnd + 1,
            "accuracy": float(acc),
            "loss": float(np.mean(round_losses)) if round_losses else 0.0,
            "eval_loss": float(eval_loss),
            "latency_round": float(t_round),
            "latency_mean": float(t_mean),
            "latency_p75": float(t_p75),
            "comm_device_to_head_mb": float(cluster_comm_mb),
            "comm_head_to_uav_mb":    float(cluster_uav_mb),
            "comm_total_mb":          float(cluster_comm_mb + cluster_uav_mb),
            "active_devices":         int(total_active),
            "cluster_times":          cluster_times_list,
        })

        print(f"  [{method:15s}] R{rnd+1:3d}/{cfg.num_rounds} "
              f"Acc={acc:.4f} Loss={eval_loss:.4f} "
              f"Lat={t_round:.3f}s Comm={cluster_comm_mb+cluster_uav_mb:.2f}MB "
              f"Active={total_active}", flush=True)

    return history
