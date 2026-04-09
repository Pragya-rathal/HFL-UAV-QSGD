from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clustering import make_clusters, select_cluster_heads, standard_topology
from compression import full_precision_bits, qsgd_quantize, topk_with_error_feedback
from config import MethodConfig, SimulationConfig
from devices import Device
from model import SmallCNN


@dataclass
class RoundStats:
    rnd: int
    accuracy: float
    loss: float
    latency: float
    latency_p75: float
    latency_max: float
    comm_mb: float
    active_devices: int


def flatten_state(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def assign_state(model: nn.Module, vec: torch.Tensor):
    pos = 0
    for p in model.parameters():
        n = p.numel()
        p.data = vec[pos : pos + n].view_as(p).data.clone()
        pos += n


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * y.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / total


def train_local(
    global_vec: torch.Tensor,
    loader: DataLoader,
    local_epochs: int,
    lr: float,
    momentum: float,
    in_channels: int,
    device: torch.device,
) -> torch.Tensor:
    model = SmallCNN(in_channels=in_channels).to(device)
    assign_state(model, global_vec.clone())
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
    return flatten_state(model).cpu() - global_vec.cpu()


def run_method(
    sim_cfg: SimulationConfig,
    method_cfg: MethodConfig,
    devices: Dict[int, Device],
    train_loaders: Dict[int, DataLoader],
    test_loader: DataLoader,
    in_channels: int,
    seed: int,
) -> pd.DataFrame:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    mc = sim_cfg.mode_config

    if method_cfg.clustered:
        heads = select_cluster_heads(devices, mc.num_clusters, sim_cfg.cluster_head_weights)
        clusters = make_clusters(devices, heads)
    else:
        heads, clusters = standard_topology(devices)

    base_model = SmallCNN(in_channels=in_channels)
    global_vec = flatten_state(base_model).cpu()
    num_params = global_vec.numel()

    residuals = {d: torch.zeros_like(global_vec) for d in devices}
    round_stats: List[RoundStats] = []

    for rnd in range(1, mc.rounds + 1):
        cluster_updates = []
        cluster_latencies = []
        total_bits = 0
        active_devices_count = 0

        for h, members in clusters.items():
            if method_cfg.quorum:
                member_scores = sorted(
                    members,
                    key=lambda m: devices[m].compute_power + devices[m].bandwidth_mbps,
                    reverse=True,
                )
                q_count = max(1, int(np.ceil(sim_cfg.quorum_fraction * len(members))))
                active_members = member_scores[:q_count]
            else:
                active_members = members

            updates = []
            device_latencies = []
            cluster_bits = 0

            for d in active_members:
                delta = train_local(
                    global_vec,
                    train_loaders[d],
                    mc.local_epochs,
                    mc.lr,
                    mc.momentum,
                    in_channels,
                    torch.device("cpu"),
                )
                if method_cfg.compression == "topk":
                    compressed, new_residual, bits = topk_with_error_feedback(delta, residuals[d], sim_cfg.topk_ratio)
                    residuals[d] = new_residual
                    sent = compressed
                elif method_cfg.compression == "qsgd":
                    sent, bits = qsgd_quantize(delta, sim_cfg.qsgd_levels)
                else:
                    sent = delta
                    bits = full_precision_bits(num_params)

                updates.append(sent)
                cluster_bits += bits

                t_comp = sim_cfg.device_base_compute_time * mc.local_epochs / devices[d].compute_power
                msg_mb = bits / (8 * 1e6)
                t_comm = (msg_mb * 8) / devices[d].bandwidth_mbps
                t_device = t_comp + t_comm
                device_latencies.append(t_device)
                devices[d].energy_joules += 0.9 * t_comp + 1.2 * t_comm

            if not updates:
                continue

            cluster_delta = torch.mean(torch.stack(updates), dim=0)
            cluster_updates.append(cluster_delta)

            upload_bits = full_precision_bits(num_params)
            cluster_bits += upload_bits
            total_bits += cluster_bits

            cluster_proc = (max(device_latencies) if device_latencies else 0.0) + sim_cfg.head_aggregation_time
            uav_tx = (upload_bits / 1e6) / sim_cfg.uav_bandwidth_mbps
            cluster_latencies.append(cluster_proc + uav_tx)

            active_devices_count += len(active_members)

        if cluster_updates:
            global_vec = global_vec + torch.mean(torch.stack(cluster_updates), dim=0)

        eval_model = SmallCNN(in_channels=in_channels)
        assign_state(eval_model, global_vec)
        acc, loss = evaluate(eval_model, test_loader, torch.device("cpu"))

        if cluster_latencies:
            lat_mean = float(np.mean(cluster_latencies))
            lat_p75 = float(np.percentile(cluster_latencies, 75))
            lat_max = float(np.max(cluster_latencies))
            lat_round = lat_max
        else:
            lat_mean = lat_p75 = lat_max = lat_round = 0.0

        round_stats.append(
            RoundStats(
                rnd=rnd,
                accuracy=acc,
                loss=loss,
                latency=lat_round,
                latency_p75=lat_p75,
                latency_max=lat_max,
                comm_mb=total_bits / (8 * 1e6),
                active_devices=active_devices_count,
            )
        )

    return pd.DataFrame([s.__dict__ for s in round_stats]).rename(columns={"rnd": "round"})
