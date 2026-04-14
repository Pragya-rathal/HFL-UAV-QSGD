"""Federated learning trainer with multiple methods."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import copy

from model import flatten_model, load_model, count_parameters, get_model_size_mb
from devices import Device
from clustering import cluster_devices
from compression import TopKCompressor, QSGDCompressor
from metrics import RoundMetrics, evaluate_model


@dataclass
class QuorumState:
    participation_history: Dict[int, int]
    last_selected: Set[int]
    
    def __init__(self, num_devices: int):
        self.participation_history = {i: -100 for i in range(num_devices)}
        self.last_selected = set()


class FederatedTrainer:
    """Hierarchical federated learning trainer."""
    
    def __init__(
        self,
        model: nn.Module,
        devices: List[Device],
        test_loader,
        config,
        clusters: Optional[Dict[int, List[int]]] = None,
        cluster_heads: Optional[Dict[int, int]] = None
    ):
        self.model = copy.deepcopy(model)
        self.global_params = flatten_model(self.model)
        self.devices = devices
        self.test_loader = test_loader
        self.config = config
        self.clusters = clusters
        self.cluster_heads = cluster_heads
        
        self.num_params = count_parameters(self.model)
        self.full_model_size_mb = get_model_size_mb(self.model)
        
        self.topk_compressor = TopKCompressor(config.compression.topk_ratio)
        self.qsgd_compressor = QSGDCompressor(config.compression.qsgd_levels)
        
        self.quorum_state = QuorumState(len(devices))
        
        for device in devices:
            device.state.residual = torch.zeros(self.num_params)
    
    def _select_quorum(
        self,
        device_ids: List[int],
        current_round: int
    ) -> List[int]:
        """Select devices for quorum with rotation guarantee."""
        if len(device_ids) <= 1:
            return device_ids
        
        cfg = self.config.quorum
        num_select = max(1, int(cfg.fraction * len(device_ids)))
        
        scores = []
        for did in device_ids:
            device = self.devices[did]
            base_score = (
                cfg.compute_weight * device.properties.compute_power +
                cfg.bandwidth_weight * device.properties.bandwidth
            )
            noise = np.random.normal(0, cfg.noise_std)
            
            rounds_since = current_round - self.quorum_state.participation_history[did]
            if rounds_since >= cfg.rotation_window:
                base_score += 10.0
            
            scores.append((did, base_score + noise))
        
        scores.sort(key=lambda x: -x[1])
        selected = [s[0] for s in scores[:num_select]]
        
        must_include = []
        for did in device_ids:
            rounds_since = current_round - self.quorum_state.participation_history[did]
            if rounds_since >= cfg.rotation_window and did not in selected:
                must_include.append(did)
        
        for did in must_include:
            if did not in selected:
                selected.append(did)
        
        for did in selected:
            self.quorum_state.participation_history[did] = current_round
        
        return selected
    
    def _aggregate_cluster_updates(
        self,
        updates: List[torch.Tensor],
        weights: List[int]
    ) -> torch.Tensor:
        """Simple mean aggregation at cluster level."""
        if len(updates) == 0:
            return torch.zeros(self.num_params)
        
        stacked = torch.stack(updates)
        return stacked.mean(dim=0)
    
    def _aggregate_global_updates(
        self,
        updates: List[torch.Tensor],
        weights: List[int]
    ) -> torch.Tensor:
        """FedAvg weighted aggregation at global level."""
        if len(updates) == 0:
            return torch.zeros(self.num_params)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return torch.stack(updates).mean(dim=0)
        
        weighted_sum = torch.zeros(self.num_params)
        for update, weight in zip(updates, weights):
            weighted_sum += (weight / total_weight) * update
        
        return weighted_sum
    
    def _compute_round_latency(
        self,
        cluster_latencies: Dict[int, float]
    ) -> float:
        """Compute total round latency as max of cluster latencies."""
        if not cluster_latencies:
            return 0.0
        return max(cluster_latencies.values())
    
    def train_round_standard(
        self,
        current_round: int,
        num_epochs: int
    ) -> RoundMetrics:
        """Method A: Standard FL (no clustering)."""
        device_ids = list(range(len(self.devices)))
        
        updates = []
        weights = []
        max_latency = 0.0
        total_comm = 0.0
        
        for did in device_ids:
            device = self.devices[did]
            update, num_samples = device.train_local(
                self.global_params.clone(),
                num_epochs
            )
            updates.append(update)
            weights.append(num_samples)
            
            _, _, latency = device.compute_latency(num_epochs, self.full_model_size_mb)
            max_latency = max(max_latency, latency)
            total_comm += self.full_model_size_mb * 2
        
        global_update = self._aggregate_global_updates(updates, weights)
        self.global_params = self.global_params + global_update
        load_model(self.model, self.global_params)
        
        accuracy, loss = evaluate_model(self.model, self.test_loader)
        
        return RoundMetrics(
            round_num=current_round,
            accuracy=accuracy,
            loss=loss,
            latency=max_latency,
            communication_mb=total_comm,
            active_devices=len(device_ids)
        )
    
    def train_round_clustered(
        self,
        current_round: int,
        num_epochs: int
    ) -> RoundMetrics:
        """Method B: Clustered FL."""
        cluster_updates = []
        cluster_weights = []
        cluster_latencies = {}
        total_comm = 0.0
        total_active = 0
        
        for cluster_id, device_ids in self.clusters.items():
            device_updates = []
            device_weights = []
            max_device_latency = 0.0
            
            for did in device_ids:
                device = self.devices[did]
                update, num_samples = device.train_local(
                    self.global_params.clone(),
                    num_epochs
                )
                device_updates.append(update)
                device_weights.append(num_samples)
                
                _, _, latency = device.compute_latency(num_epochs, self.full_model_size_mb)
                max_device_latency = max(max_device_latency, latency)
                total_comm += self.full_model_size_mb
            
            cluster_update = self._aggregate_cluster_updates(device_updates, device_weights)
            cluster_updates.append(cluster_update)
            cluster_weights.append(sum(device_weights))
            cluster_latencies[cluster_id] = max_device_latency
            total_active += len(device_ids)
            
            total_comm += self.full_model_size_mb
        
        global_update = self._aggregate_global_updates(cluster_updates, cluster_weights)
        self.global_params = self.global_params + global_update
        load_model(self.model, self.global_params)
        
        accuracy, loss = evaluate_model(self.model, self.test_loader)
        round_latency = self._compute_round_latency(cluster_latencies)
        
        return RoundMetrics(
            round_num=current_round,
            accuracy=accuracy,
            loss=loss,
            latency=round_latency,
            communication_mb=total_comm,
            active_devices=total_active,
            cluster_latencies=cluster_latencies
        )
    
    def train_round_topk(
        self,
        current_round: int,
        num_epochs: int
    ) -> RoundMetrics:
        """Method C: Clustered FL + Top-K + Error Feedback."""
        cluster_updates = []
        cluster_weights = []
        cluster_latencies = {}
        total_comm = 0.0
        total_active = 0
        
        topk_size_mb = self.topk_compressor.get_communication_mb(self.num_params)
        
        for cluster_id, device_ids in self.clusters.items():
            device_updates = []
            device_weights = []
            max_device_latency = 0.0
            
            for did in device_ids:
                device = self.devices[did]
                update, num_samples = device.train_local(
                    self.global_params.clone(),
                    num_epochs
                )
                
                compressed, new_residual, _ = self.topk_compressor.compress(
                    update,
                    device.state.residual
                )
                device.state.residual = new_residual
                
                device_updates.append(compressed)
                device_weights.append(num_samples)
                
                _, _, latency = device.compute_latency(num_epochs, topk_size_mb)
                max_device_latency = max(max_device_latency, latency)
                total_comm += topk_size_mb
            
            cluster_update = self._aggregate_cluster_updates(device_updates, device_weights)
            cluster_updates.append(cluster_update)
            cluster_weights.append(sum(device_weights))
            cluster_latencies[cluster_id] = max_device_latency
            total_active += len(device_ids)
            
            total_comm += topk_size_mb
        
        global_update = self._aggregate_global_updates(cluster_updates, cluster_weights)
        self.global_params = self.global_params + global_update
        load_model(self.model, self.global_params)
        
        accuracy, loss = evaluate_model(self.model, self.test_loader)
        round_latency = self._compute_round_latency(cluster_latencies)
        
        return RoundMetrics(
            round_num=current_round,
            accuracy=accuracy,
            loss=loss,
            latency=round_latency,
            communication_mb=total_comm,
            active_devices=total_active,
            cluster_latencies=cluster_latencies
        )
    
    def train_round_qsgd(
        self,
        current_round: int,
        num_epochs: int
    ) -> RoundMetrics:
        """Method D: Clustered FL + QSGD."""
        cluster_updates = []
        cluster_weights = []
        cluster_latencies = {}
        total_comm = 0.0
        total_active = 0
        
        qsgd_size_mb = self.qsgd_compressor.get_communication_mb(self.num_params)
        
        for cluster_id, device_ids in self.clusters.items():
            device_updates = []
            device_weights = []
            max_device_latency = 0.0
            
            for did in device_ids:
                device = self.devices[did]
                update, num_samples = device.train_local(
                    self.global_params.clone(),
                    num_epochs
                )
                
                quantized, signs, norm = self.qsgd_compressor.compress(update)
                decompressed = self.qsgd_compressor.decompress(quantized, signs, norm)
                
                device_updates.append(decompressed)
                device_weights.append(num_samples)
                
                _, _, latency = device.compute_latency(num_epochs, qsgd_size_mb)
                max_device_latency = max(max_device_latency, latency)
                total_comm += qsgd_size_mb
            
            cluster_update = self._aggregate_cluster_updates(device_updates, device_weights)
            cluster_updates.append(cluster_update)
            cluster_weights.append(sum(device_weights))
            cluster_latencies[cluster_id] = max_device_latency
            total_active += len(device_ids)
            
            total_comm += qsgd_size_mb
        
        global_update = self._aggregate_global_updates(cluster_updates, cluster_weights)
        self.global_params = self.global_params + global_update
        load_model(self.model, self.global_params)
        
        accuracy, loss = evaluate_model(self.model, self.test_loader)
        round_latency = self._compute_round_latency(cluster_latencies)
        
        return RoundMetrics(
            round_num=current_round,
            accuracy=accuracy,
            loss=loss,
            latency=round_latency,
            communication_mb=total_comm,
            active_devices=total_active,
            cluster_latencies=cluster_latencies
        )
    
    def train_round_topk_quorum(
        self,
        current_round: int,
        num_epochs: int
    ) -> RoundMetrics:
        """Method E: Clustered FL + Top-K + Quorum."""
        cluster_updates = []
        cluster_weights = []
        cluster_latencies = {}
        total_comm = 0.0
        total_active = 0
        
        topk_size_mb = self.topk_compressor.get_communication_mb(self.num_params)
        
        for cluster_id, device_ids in self.clusters.items():
            selected_ids = self._select_quorum(device_ids, current_round)
            
            device_updates = []
            device_weights = []
            max_device_latency = 0.0
            
            for did in selected_ids:
                device = self.devices[did]
                update, num_samples = device.train_local(
                    self.global_params.clone(),
                    num_epochs
                )
                
                compressed, new_residual, _ = self.topk_compressor.compress(
                    update,
                    device.state.residual
                )
                device.state.residual = new_residual
                
                device_updates.append(compressed)
                device_weights.append(num_samples)
                
                _, _, latency = device.compute_latency(num_epochs, topk_size_mb)
                max_device_latency = max(max_device_latency, latency)
                total_comm += topk_size_mb
            
            if device_updates:
                cluster_update = self._aggregate_cluster_updates(device_updates, device_weights)
                cluster_updates.append(cluster_update)
                cluster_weights.append(sum(device_weights))
                cluster_latencies[cluster_id] = max_device_latency
                total_active += len(selected_ids)
                
                total_comm += topk_size_mb
        
        if cluster_updates:
            global_update = self._aggregate_global_updates(cluster_updates, cluster_weights)
            self.global_params = self.global_params + global_update
        
        load_model(self.model, self.global_params)
        
        accuracy, loss = evaluate_model(self.model, self.test_loader)
        round_latency = self._compute_round_latency(cluster_latencies)
        
        return RoundMetrics(
            round_num=current_round,
            accuracy=accuracy,
            loss=loss,
            latency=round_latency,
            communication_mb=total_comm,
            active_devices=total_active,
            cluster_latencies=cluster_latencies
        )
    
    def train_round_qsgd_quorum(
        self,
        current_round: int,
        num_epochs: int
    ) -> RoundMetrics:
        """Method F: Clustered FL + QSGD + Quorum."""
        cluster_updates = []
        cluster_weights = []
        cluster_latencies = {}
        total_comm = 0.0
        total_active = 0
        
        qsgd_size_mb = self.qsgd_compressor.get_communication_mb(self.num_params)
        
        for cluster_id, device_ids in self.clusters.items():
            selected_ids = self._select_quorum(device_ids, current_round)
            
            device_updates = []
            device_weights = []
            max_device_latency = 0.0
            
            for did in selected_ids:
                device = self.devices[did]
                update, num_samples = device.train_local(
                    self.global_params.clone(),
                    num_epochs
                )
                
                quantized, signs, norm = self.qsgd_compressor.compress(update)
                decompressed = self.qsgd_compressor.decompress(quantized, signs, norm)
                
                device_updates.append(decompressed)
                device_weights.append(num_samples)
                
                _, _, latency = device.compute_latency(num_epochs, qsgd_size_mb)
                max_device_latency = max(max_device_latency, latency)
                total_comm += qsgd_size_mb
            
            if device_updates:
                cluster_update = self._aggregate_cluster_updates(device_updates, device_weights)
                cluster_updates.append(cluster_update)
                cluster_weights.append(sum(device_weights))
                cluster_latencies[cluster_id] = max_device_latency
                total_active += len(selected_ids)
                
                total_comm += qsgd_size_mb
        
        if cluster_updates:
            global_update = self._aggregate_global_updates(cluster_updates, cluster_weights)
            self.global_params = self.global_params + global_update
        
        load_model(self.model, self.global_params)
        
        accuracy, loss = evaluate_model(self.model, self.test_loader)
        round_latency = self._compute_round_latency(cluster_latencies)
        
        return RoundMetrics(
            round_num=current_round,
            accuracy=accuracy,
            loss=loss,
            latency=round_latency,
            communication_mb=total_comm,
            active_devices=total_active,
            cluster_latencies=cluster_latencies
        )
    
    def reset(self, initial_model: nn.Module):
        """Reset trainer for new experiment."""
        self.model = copy.deepcopy(initial_model)
        self.global_params = flatten_model(self.model)
        self.quorum_state = QuorumState(len(self.devices))
        
        for device in self.devices:
            device.state.residual = torch.zeros(self.num_params)
            device.state.last_participated_round = -1
