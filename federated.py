import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clustering import choose_cluster_heads, form_clusters, select_quorum
from compression import full_precision_size_mb, qsgd_quantize, topk_compress_with_error_feedback
from config import ExperimentConfig, METHOD_MAP
from devices import DeviceProfile, get_model_delta, weighted_average_deltas
from metrics import RoundMetric
from model import CNNClassifier, count_parameters


@dataclass
class MethodFlags:
    clustered: bool
    compression: str  # none|topk|qsgd
    quorum: bool


METHOD_FLAGS = {
    "A": MethodFlags(clustered=False, compression="none", quorum=False),
    "B": MethodFlags(clustered=True, compression="none", quorum=False),
    "C": MethodFlags(clustered=True, compression="topk", quorum=False),
    "D": MethodFlags(clustered=True, compression="qsgd", quorum=False),
    "E": MethodFlags(clustered=True, compression="topk", quorum=True),
    "F": MethodFlags(clustered=True, compression="qsgd", quorum=True),
}


class FederatedRunner:
    def __init__(
        self,
        cfg: ExperimentConfig,
        device_loaders: Dict[int, DataLoader],
        test_loader: DataLoader,
        in_channels: int,
        image_size: int,
        device_profiles: Dict[int, DeviceProfile],
        cluster_map: Dict[int, List[int]],
    ):
        self.cfg = cfg
        self.device_loaders = device_loaders
        self.test_loader = test_loader
        self.in_channels = in_channels
        self.image_size = image_size
        self.device_profiles = device_profiles
        self.cluster_map = cluster_map
        self.cluster_heads = choose_cluster_heads(cluster_map, device_profiles, cfg.system.head_score_weights)

    def _build_model(self, seed: int):
        torch.manual_seed(seed)
        model = CNNClassifier(self.in_channels, num_classes=self.cfg.data.num_classes, image_size=self.image_size)
        return model

    def _train_local(self, base_model: nn.Module, loader: DataLoader, seed: int) -> Tuple[nn.Module, float]:
        model = copy.deepcopy(base_model)
        model.train()
        torch.manual_seed(seed)

        if self.cfg.train.optimizer == "sgd":
            opt = torch.optim.SGD(
                model.parameters(),
                lr=self.cfg.train.lr,
                momentum=self.cfg.train.momentum,
                weight_decay=self.cfg.train.weight_decay,
            )
        else:
            opt = torch.optim.Adam(model.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

        criterion = nn.CrossEntropyLoss()
        losses = []
        for _ in range(self.cfg.train.local_epochs):
            for x, y in loader:
                opt.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        return model, float(np.mean(losses)) if losses else 0.0

    @torch.no_grad()
    def _evaluate(self, model: nn.Module) -> float:
        model.eval()
        correct = 0
        total = 0
        for x, y in self.test_loader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / total if total else 0.0

    def _compute_latency_energy(
        self,
        num_samples: int,
        num_batches: int,
        model_params: int,
        compute_power: float,
        bandwidth_mbps: float,
        msg_size_mb: float,
    ) -> Tuple[float, float, float]:
        base_compute = (
            self.cfg.train.local_epochs
            * num_batches
            * num_samples
            * model_params
            * self.cfg.system.base_time_per_sample_per_param
        )
        t_comp = base_compute / compute_power
        t_comm = msg_size_mb * 8.0 / max(bandwidth_mbps, 1e-6)
        energy = self.cfg.system.energy_compute_coeff * t_comp * compute_power + self.cfg.system.energy_tx_coeff * t_comm
        return t_comp, t_comm, energy

    def _cluster_aggregate(self, updates, weights):
        if len(updates) == 1:
            return updates[0]
        return weighted_average_deltas(updates, weights)

    def run_method(self, method_key: str, seed: int) -> List[RoundMetric]:
        flags = METHOD_FLAGS[method_key]
        rng = np.random.default_rng(seed + 1000)
        model = self._build_model(seed)
        global_state = copy.deepcopy(model.state_dict())
        model_params = count_parameters(model)
        full_msg_mb = full_precision_size_mb(model_params, 32)

        for dev in self.device_profiles.values():
            dev.init_residual(global_state)

        method_name = METHOD_MAP[method_key]
        all_metrics: List[RoundMetric] = []

        for r in range(1, self.cfg.train.rounds + 1):
            round_cluster_payloads = []
            round_cluster_weights = []
            round_latency_components = []
            round_comm_mb = 0.0
            round_energy = 0.0
            round_losses = []
            active_devices = 0
            round_comp_ratios = []

            if not flags.clustered:
                clusters = {0: list(self.device_profiles.keys())}
            else:
                clusters = self.cluster_map

            for cid, members in clusters.items():
                participants = members
                if flags.quorum:
                    participants = select_quorum(
                        members,
                        self.device_profiles,
                        self.cfg.quorum_fraction,
                        self.cfg.quorum_count,
                        self.cfg.system.quorum_score_weights,
                        self.cfg.system.random_quorum_jitter,
                        rng,
                    )

                device_updates = []
                device_weights = []
                device_times = []
                for dev_id in participants:
                    active_devices += 1
                    local_model, loss = self._train_local(model, self.device_loaders[dev_id], seed + r * 37 + dev_id)
                    round_losses.append(loss)
                    delta = get_model_delta(local_model, global_state)

                    if flags.compression == "topk":
                        compressed, new_res, msg_mb, cr = topk_compress_with_error_feedback(
                            delta,
                            self.device_profiles[dev_id].residual,
                            self.cfg.compression.topk_ratio,
                        )
                        self.device_profiles[dev_id].residual = new_res
                        sent_delta = compressed
                        round_comp_ratios.append(cr)
                    elif flags.compression == "qsgd":
                        sent_delta, msg_mb, cr = qsgd_quantize(delta, self.cfg.compression.qsgd_levels)
                        round_comp_ratios.append(cr)
                    else:
                        sent_delta = delta
                        msg_mb = full_msg_mb
                        round_comp_ratios.append(1.0)

                    n_samples = len(self.device_loaders[dev_id].dataset)
                    n_batches = len(self.device_loaders[dev_id])
                    t_comp, t_comm, e = self._compute_latency_energy(
                        num_samples=n_samples,
                        num_batches=n_batches,
                        model_params=model_params,
                        compute_power=self.device_profiles[dev_id].compute_power,
                        bandwidth_mbps=self.device_profiles[dev_id].bandwidth_mbps,
                        msg_size_mb=msg_mb,
                    )
                    round_energy += e

                    d_time = t_comp + t_comm
                    device_times.append(d_time)
                    round_comm_mb += msg_mb
                    device_updates.append(sent_delta)
                    device_weights.append(n_samples)

                if len(device_updates) == 0:
                    continue
                c_delta = self._cluster_aggregate(device_updates, device_weights)

                agg_time = len(device_updates) * model_params * self.cfg.system.head_agg_time_per_param_per_update
                ch_to_uav_mb = full_msg_mb
                ch_to_uav_time = ch_to_uav_mb * 8.0 / self.cfg.system.uav_link_bandwidth_mbps

                round_comm_mb += ch_to_uav_mb
                cluster_latency = max(device_times) + agg_time + ch_to_uav_time
                round_latency_components.append(cluster_latency)

                round_cluster_payloads.append(c_delta)
                round_cluster_weights.append(sum(device_weights))

            if len(round_cluster_payloads) == 0:
                continue

            g_delta = self._cluster_aggregate(round_cluster_payloads, round_cluster_weights)
            global_state = {k: global_state[k] + g_delta[k] for k in global_state}
            model.load_state_dict(global_state)

            acc = self._evaluate(model)
            lat = float(max(round_latency_components)) if round_latency_components else 0.0
            p75 = float(np.percentile(round_latency_components, 75)) if round_latency_components else 0.0
            mlat = float(np.max(round_latency_components)) if round_latency_components else 0.0

            all_metrics.append(
                RoundMetric(
                    round_id=r,
                    seed=seed,
                    method=method_name,
                    train_loss=float(np.mean(round_losses)) if round_losses else 0.0,
                    test_accuracy=acc,
                    comm_mb=round_comm_mb,
                    latency_s=lat,
                    active_devices=active_devices,
                    energy_j=round_energy,
                    p75_latency_s=p75,
                    max_latency_s=mlat,
                    compression_ratio=float(np.mean(round_comp_ratios)) if round_comp_ratios else 1.0,
                )
            )

        return all_metrics


def create_device_profiles(cfg: ExperimentConfig, seed: int) -> Dict[int, DeviceProfile]:
    rng = np.random.default_rng(seed)
    devs = {}
    for i in range(cfg.num_devices):
        cp = rng.uniform(*cfg.system.compute_power_range)
        bw = rng.uniform(*cfg.system.bandwidth_range_mbps)
        dist = rng.uniform(*cfg.system.distance_range_m)
        cq = 1.0 / dist
        cc = rng.uniform(*cfg.system.clustering_coeff_range)
        devs[i] = DeviceProfile(
            device_id=i,
            compute_power=float(cp),
            bandwidth_mbps=float(bw),
            distance_m=float(dist),
            channel_quality=float(cq),
            clustering_coefficient=float(cc),
        )
    return devs


def ensure_dirs(cfg: ExperimentConfig):
    os.makedirs(cfg.output_root, exist_ok=True)
