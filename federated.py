### federated.py
import numpy as np
import torch
import copy
from model import flatten_model, load_model, create_model, count_parameters
from compression import (
    topk_compress, topk_message_size_MB,
    qsgd_quantize, qsgd_dequantize, qsgd_message_size_MB,
    full_precision_size_MB, select_quorum
)
from clustering import cluster_devices, get_cluster_head


class FederatedTrainer:
    def __init__(self, config, devices, test_loader):
        self.config = config
        self.devices = devices
        self.test_loader = test_loader
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Global model stored ONLY here
        self.global_model = create_model(config).to(self.torch_device)
        self.global_flat = flatten_model(self.global_model).cpu()
        self.num_params = len(self.global_flat)

        # Initialize residuals for error feedback
        for d in devices:
            d.reset_residual(self.num_params)

        # Round tracking
        self.current_round = 0
        self.rng = np.random.RandomState(config['seed'])

    def _get_global_copy(self):
        """Return a copy of the current global flat tensor."""
        return self.global_flat.clone()

    def _evaluate(self):
        model = create_model(self.config).to(self.torch_device)
        load_model(model, self.global_flat.to(self.torch_device))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.torch_device), y.to(self.torch_device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total

    # ─────────────────────────────────────────
    # Method A: Standard FL (no clustering)
    # ─────────────────────────────────────────
    def run_standard_fl(self, num_rounds):
        history = []
        all_ids = list(range(len(self.devices)))

        for rnd in range(num_rounds):
            global_copy = self._get_global_copy()
            updates = []
            sizes = []
            latencies = []
            total_comm = 0.0
            total_loss = 0.0

            for did in all_ids:
                d = self.devices[did]
                update, loss, num_batches = d.train_local(global_copy, self.config['local_epochs'])
                updates.append(update)
                sizes.append(d.dataset_size)
                total_loss += loss

                msg_MB = full_precision_size_MB(self.num_params)
                total_comm += msg_MB
                lat = d.compute_latency(num_batches, self.config['local_epochs'], msg_MB)
                latencies.append(lat)

            # FedAvg aggregation
            total_size = sum(sizes)
            agg_update = torch.zeros(self.num_params)
            for upd, sz in zip(updates, sizes):
                agg_update += upd * (sz / total_size)
            self.global_flat = global_copy + agg_update

            acc = self._evaluate()
            avg_loss = total_loss / len(all_ids)
            round_latency = max(latencies) if latencies else 0.0

            history.append({
                'round': rnd + 1,
                'accuracy': acc,
                'loss': avg_loss,
                'latency': round_latency,
                'communication_MB': total_comm,
                'active_devices': len(all_ids)
            })
            print(f"[A] Round {rnd+1}/{num_rounds} | Acc={acc:.4f} Loss={avg_loss:.4f} "
                  f"Lat={round_latency:.3f}s Comm={total_comm:.2f}MB")
        return history

    # ─────────────────────────────────────────
    # Method B: Clustered FL
    # ─────────────────────────────────────────
    def run_clustered_fl(self, num_rounds):
        clusters = cluster_devices(self.devices, self.config['num_clusters'], seed=self.config['seed'])
        history = []

        for rnd in range(num_rounds):
            global_copy = self._get_global_copy()
            cluster_updates = []
            cluster_sizes = []
            cluster_latencies = []
            total_comm = 0.0
            total_loss = 0.0
            active_count = 0

            for cluster in clusters:
                device_updates = []
                device_sizes = []
                device_latencies = []

                for did in cluster:
                    d = self.devices[did]
                    update, loss, num_batches = d.train_local(global_copy, self.config['local_epochs'])
                    device_updates.append(update)
                    device_sizes.append(d.dataset_size)
                    total_loss += loss
                    active_count += 1

                    msg_MB = full_precision_size_MB(self.num_params)
                    total_comm += msg_MB
                    lat = d.compute_latency(num_batches, self.config['local_epochs'], msg_MB)
                    device_latencies.append(lat)

                # Cluster aggregation: simple mean of updates
                cluster_update = torch.stack(device_updates).mean(dim=0)
                cluster_size = sum(device_sizes)
                cluster_updates.append(cluster_update)
                cluster_sizes.append(cluster_size)
                cluster_latencies.append(max(device_latencies) if device_latencies else 0.0)

            # UAV aggregation: FedAvg weighted by dataset size
            total_size = sum(cluster_sizes)
            agg_update = torch.zeros(self.num_params)
            for upd, sz in zip(cluster_updates, cluster_sizes):
                agg_update += upd * (sz / total_size)
            self.global_flat = global_copy + agg_update

            acc = self._evaluate()
            avg_loss = total_loss / max(1, active_count)
            round_latency = max(cluster_latencies) if cluster_latencies else 0.0

            history.append({
                'round': rnd + 1,
                'accuracy': acc,
                'loss': avg_loss,
                'latency': round_latency,
                'communication_MB': total_comm,
                'active_devices': active_count
            })
            print(f"[B] Round {rnd+1}/{num_rounds} | Acc={acc:.4f} Loss={avg_loss:.4f} "
                  f"Lat={round_latency:.3f}s Comm={total_comm:.2f}MB")
        return history

    # ─────────────────────────────────────────
    # Method C: Cluster + Top-K + Error Feedback
    # ─────────────────────────────────────────
    def run_clustered_topk(self, num_rounds):
        clusters = cluster_devices(self.devices, self.config['num_clusters'], seed=self.config['seed'])
        ratio = self.config['topk_ratio']
        history = []

        # Reset residuals
        for d in self.devices:
            d.reset_residual(self.num_params)

        for rnd in range(num_rounds):
            global_copy = self._get_global_copy()
            cluster_updates = []
            cluster_sizes = []
            cluster_latencies = []
            total_comm = 0.0
            total_loss = 0.0
            active_count = 0

            for cluster in clusters:
                device_updates = []
                device_sizes = []
                device_latencies = []

                for did in cluster:
                    d = self.devices[did]
                    update, loss, num_batches = d.train_local(global_copy, self.config['local_epochs'])
                    total_loss += loss
                    active_count += 1

                    # Top-K with error feedback
                    compressed, d.residual, k = topk_compress(update, d.residual, ratio)
                    device_updates.append(compressed)
                    device_sizes.append(d.dataset_size)

                    msg_MB = topk_message_size_MB(k)
                    total_comm += msg_MB
                    lat = d.compute_latency(num_batches, self.config['local_epochs'], msg_MB)
                    device_latencies.append(lat)

                cluster_update = torch.stack(device_updates).mean(dim=0)
                cluster_size = sum(device_sizes)
                cluster_updates.append(cluster_update)
                cluster_sizes.append(cluster_size)
                cluster_latencies.append(max(device_latencies) if device_latencies else 0.0)

            total_size = sum(cluster_sizes)
            agg_update = torch.zeros(self.num_params)
            for upd, sz in zip(cluster_updates, cluster_sizes):
                agg_update += upd * (sz / total_size)
            self.global_flat = global_copy + agg_update

            acc = self._evaluate()
            avg_loss = total_loss / max(1, active_count)
            round_latency = max(cluster_latencies) if cluster_latencies else 0.0

            history.append({
                'round': rnd + 1,
                'accuracy': acc,
                'loss': avg_loss,
                'latency': round_latency,
                'communication_MB': total_comm,
                'active_devices': active_count
            })
            print(f"[C] Round {rnd+1}/{num_rounds} | Acc={acc:.4f} Loss={avg_loss:.4f} "
                  f"Lat={round_latency:.3f}s Comm={total_comm:.2f}MB")
        return history

    # ─────────────────────────────────────────
    # Method D: Cluster + QSGD
    # ─────────────────────────────────────────
    def run_clustered_qsgd(self, num_rounds):
        clusters = cluster_devices(self.devices, self.config['num_clusters'], seed=self.config['seed'])
        s = self.config['qsgd_levels']
        history = []

        for rnd in range(num_rounds):
            global_copy = self._get_global_copy()
            cluster_updates = []
            cluster_sizes = []
            cluster_latencies = []
            total_comm = 0.0
            total_loss = 0.0
            active_count = 0

            for cluster in clusters:
                device_updates = []
                device_sizes = []
                device_latencies = []

                for did in cluster:
                    d = self.devices[did]
                    update, loss, num_batches = d.train_local(global_copy, self.config['local_epochs'])
                    total_loss += loss
                    active_count += 1

                    # QSGD quantization
                    q, signs, norm, _ = qsgd_quantize(update, s)
                    dequant = qsgd_dequantize(q, signs, norm, s)
                    device_updates.append(dequant)
                    device_sizes.append(d.dataset_size)

                    msg_MB = qsgd_message_size_MB(self.num_params, s)
                    total_comm += msg_MB
                    lat = d.compute_latency(num_batches, self.config['local_epochs'], msg_MB)
                    device_latencies.append(lat)

                cluster_update = torch.stack(device_updates).mean(dim=0)
                cluster_size = sum(device_sizes)
                cluster_updates.append(cluster_update)
                cluster_sizes.append(cluster_size)
                cluster_latencies.append(max(device_latencies) if device_latencies else 0.0)

            total_size = sum(cluster_sizes)
            agg_update = torch.zeros(self.num_params)
            for upd, sz in zip(cluster_updates, cluster_sizes):
                agg_update += upd * (sz / total_size)
            self.global_flat = global_copy + agg_update

            acc = self._evaluate()
            avg_loss = total_loss / max(1, active_count)
            round_latency = max(cluster_latencies) if cluster_latencies else 0.0

            history.append({
                'round': rnd + 1,
                'accuracy': acc,
                'loss': avg_loss,
                'latency': round_latency,
                'communication_MB': total_comm,
                'active_devices': active_count
            })
            print(f"[D] Round {rnd+1}/{num_rounds} | Acc={acc:.4f} Loss={avg_loss:.4f} "
                  f"Lat={round_latency:.3f}s Comm={total_comm:.2f}MB")
        return history

    # ─────────────────────────────────────────
    # Method E: Cluster + Top-K + Quorum
    # ─────────────────────────────────────────
    def run_clustered_topk_quorum(self, num_rounds):
        clusters = cluster_devices(self.devices, self.config['num_clusters'], seed=self.config['seed'])
        ratio = self.config['topk_ratio']
        fraction = self.config['quorum_fraction']
        K = self.config['quorum_K']
        history = []

        for d in self.devices:
            d.reset_residual(self.num_params)

        for rnd in range(num_rounds):
            global_copy = self._get_global_copy()
            cluster_updates = []
            cluster_sizes = []
            cluster_latencies = []
            total_comm = 0.0
            total_loss = 0.0
            active_count = 0

            for cluster in clusters:
                selected = select_quorum(self.devices, cluster, fraction, K, rnd, rng=self.rng)
                device_updates = []
                device_sizes = []
                device_latencies = []

                for did in selected:
                    d = self.devices[did]
                    update, loss, num_batches = d.train_local(global_copy, self.config['local_epochs'])
                    d.last_participated = rnd
                    total_loss += loss
                    active_count += 1

                    compressed, d.residual, k = topk_compress(update, d.residual, ratio)
                    device_updates.append(compressed)
                    device_sizes.append(d.dataset_size)

                    msg_MB = topk_message_size_MB(k)
                    total_comm += msg_MB
                    lat = d.compute_latency(num_batches, self.config['local_epochs'], msg_MB)
                    device_latencies.append(lat)

                cluster_update = torch.stack(device_updates).mean(dim=0)
                cluster_size = sum(device_sizes)
                cluster_updates.append(cluster_update)
                cluster_sizes.append(cluster_size)
                cluster_latencies.append(max(device_latencies) if device_latencies else 0.0)

            total_size = sum(cluster_sizes)
            agg_update = torch.zeros(self.num_params)
            for upd, sz in zip(cluster_updates, cluster_sizes):
                agg_update += upd * (sz / total_size)
            self.global_flat = global_copy + agg_update

            acc = self._evaluate()
            avg_loss = total_loss / max(1, active_count)
            round_latency = max(cluster_latencies) if cluster_latencies else 0.0

            history.append({
                'round': rnd + 1,
                'accuracy': acc,
                'loss': avg_loss,
                'latency': round_latency,
                'communication_MB': total_comm,
                'active_devices': active_count
            })
            print(f"[E] Round {rnd+1}/{num_rounds} | Acc={acc:.4f} Loss={avg_loss:.4f} "
                  f"Lat={round_latency:.3f}s Comm={total_comm:.2f}MB")
        return history

    # ─────────────────────────────────────────
    # Method F: Cluster + QSGD + Quorum
    # ─────────────────────────────────────────
    def run_clustered_qsgd_quorum(self, num_rounds):
        clusters = cluster_devices(self.devices, self.config['num_clusters'], seed=self.config['seed'])
        s = self.config['qsgd_levels']
        fraction = self.config['quorum_fraction']
        K = self.config['quorum_K']
        history = []

        for rnd in range(num_rounds):
            global_copy = self._get_global_copy()
            cluster_updates = []
            cluster_sizes = []
            cluster_latencies = []
            total_comm = 0.0
            total_loss = 0.0
            active_count = 0

            for cluster in clusters:
                selected = select_quorum(self.devices, cluster, fraction, K, rnd, rng=self.rng)
                device_updates = []
                device_sizes = []
                device_latencies = []

                for did in selected:
                    d = self.devices[did]
                    update, loss, num_batches = d.train_local(global_copy, self.config['local_epochs'])
                    d.last_participated = rnd
                    total_loss += loss
                    active_count += 1

                    q, signs, norm, _ = qsgd_quantize(update, s)
                    dequant = qsgd_dequantize(q, signs, norm, s)
                    device_updates.append(dequant)
                    device_sizes.append(d.dataset_size)

                    msg_MB = qsgd_message_size_MB(self.num_params, s)
                    total_comm += msg_MB
                    lat = d.compute_latency(num_batches, self.config['local_epochs'], msg_MB)
                    device_latencies.append(lat)

                cluster_update = torch.stack(device_updates).mean(dim=0)
                cluster_size = sum(device_sizes)
                cluster_updates.append(cluster_update)
                cluster_sizes.append(cluster_size)
                cluster_latencies.append(max(device_latencies) if device_latencies else 0.0)

            total_size = sum(cluster_sizes)
            agg_update = torch.zeros(self.num_params)
            for upd, sz in zip(cluster_updates, cluster_sizes):
                agg_update += upd * (sz / total_size)
            self.global_flat = global_copy + agg_update

            acc = self._evaluate()
            avg_loss = total_loss / max(1, active_count)
            round_latency = max(cluster_latencies) if cluster_latencies else 0.0

            history.append({
                'round': rnd + 1,
                'accuracy': acc,
                'loss': avg_loss,
                'latency': round_latency,
                'communication_MB': total_comm,
                'active_devices': active_count
            })
            print(f"[F] Round {rnd+1}/{num_rounds} | Acc={acc:.4f} Loss={avg_loss:.4f} "
                  f"Lat={round_latency:.3f}s Comm={total_comm:.2f}MB")
        return history

    def reset_global_model(self):
        """Reset global model to random initialization for fair comparison."""
        self.global_model = create_model(self.config).to(self.torch_device)
        self.global_flat = flatten_model(self.global_model).cpu()
        for d in self.devices:
            d.reset_residual(self.num_params)
            d.last_participated = -1
