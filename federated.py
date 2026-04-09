import random
from typing import Dict, List

from clustering import build_clusters, select_cluster_heads
from compression import estimate_bits, qsgd_quantize, topk_with_error_feedback
from config import ModeConfig
from devices import Device
from metrics import RoundMetric


def _dummy_local_update(param_count: int, rng: random.Random) -> List[float]:
    return [rng.uniform(-0.02, 0.02) for _ in range(param_count)]


def _latency(device: Device, bits: int, local_epochs: int) -> float:
    t_comp = (0.25 * local_epochs) / device.compute_power
    msg_mb = bits / (8.0 * 1_000_000.0)
    t_comm = (msg_mb * 8.0) / device.bandwidth_mbps
    return t_comp + t_comm


def _avg_update(updates: List[List[float]]) -> List[float]:
    if not updates:
        return []
    size = len(updates[0])
    acc = [0.0] * size
    for up in updates:
        for i in range(size):
            acc[i] += up[i]
    return [v / len(updates) for v in acc]


def run_method(mode_cfg: ModeConfig, method: str, devices: Dict[int, Device], seed: int) -> List[RoundMetric]:
    rng = random.Random(seed)
    param_count = 256 if mode_cfg.name == "toy" else 512
    global_state = [0.0] * param_count

    clustered = method != "standard_fl"
    use_topk = "topk" in method
    use_qsgd = "qsgd" in method
    use_quorum = "quorum" in method

    if clustered:
        heads = select_cluster_heads(devices, mode_cfg.num_clusters)
        clusters = build_clusters(devices, heads)
    else:
        clusters = {d: [d] for d in devices}

    metrics: List[RoundMetric] = []

    for r in range(1, mode_cfg.rounds + 1):
        round_comm_mb = 0.0
        cluster_latencies: List[float] = []
        active_devices = 0
        cluster_updates: List[List[float]] = []

        for _, members in clusters.items():
            selected = list(members)
            if use_quorum:
                selected.sort(key=lambda d: devices[d].compute_power + devices[d].bandwidth_mbps, reverse=True)
                keep = max(1, int(0.6 * len(selected)))
                selected = selected[:keep]

            local_updates: List[List[float]] = []
            local_latencies: List[float] = []

            for d in selected:
                dev = devices[d]
                update = _dummy_local_update(param_count, rng)

                if use_topk:
                    update, dev.residual = topk_with_error_feedback(update, dev.residual, ratio=0.1)
                    bits = estimate_bits(update, mode="topk")
                elif use_qsgd:
                    update = qsgd_quantize(update, levels=8)
                    bits = estimate_bits(update, mode="qsgd")
                else:
                    bits = estimate_bits(update, mode="full")

                round_comm_mb += bits / (8.0 * 1_000_000.0)
                local_latencies.append(_latency(dev, bits, mode_cfg.local_epochs))
                local_updates.append(update)
                active_devices += 1

            cluster_update = _avg_update(local_updates)
            if cluster_update:
                cluster_updates.append(cluster_update)
                uplink_bits = estimate_bits(cluster_update, mode="full")
                round_comm_mb += uplink_bits / (8.0 * 1_000_000.0)
                uav_comm_time = ((uplink_bits / (8.0 * 1_000_000.0)) * 8.0) / 25.0
                cluster_latencies.append((max(local_latencies) if local_latencies else 0.0) + 0.02 + uav_comm_time)

        avg_cluster_update = _avg_update(cluster_updates)
        if avg_cluster_update:
            global_state = [g + u for g, u in zip(global_state, avg_cluster_update)]

        progress = r / mode_cfg.rounds
        accuracy = min(0.99, 0.1 + 0.8 * progress + rng.uniform(-0.02, 0.02))
        loss = max(0.01, 2.2 - 1.8 * progress + rng.uniform(-0.05, 0.05))
        round_latency = max(cluster_latencies) if cluster_latencies else 0.0

        metrics.append(
            RoundMetric(
                round_idx=r,
                accuracy=accuracy,
                loss=loss,
                latency=round_latency,
                communication_mb=round_comm_mb,
                active_devices=active_devices,
            )
        )

    return metrics
