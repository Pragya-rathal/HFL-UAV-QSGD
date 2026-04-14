"""Compression methods: Top-K with error feedback and QSGD."""

import torch
import numpy as np
from typing import Tuple, Optional
import math


class TopKCompressor:
    """Top-K sparsification with error feedback."""
    
    def __init__(self, ratio: float = 0.1):
        self.ratio = ratio
    
    def compress(
        self,
        update: torch.Tensor,
        residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress update using Top-K sparsification.
        
        Returns:
            compressed: sparse tensor with only top-k values
            new_residual: u - compressed for error feedback
            indices: indices of top-k elements
        """
        if residual is not None:
            u = update + residual
        else:
            u = update.clone()
        
        k = max(1, int(self.ratio * len(u)))
        
        abs_u = torch.abs(u)
        _, indices = torch.topk(abs_u, k)
        
        compressed = torch.zeros_like(u)
        compressed[indices] = u[indices]
        
        new_residual = u - compressed
        
        return compressed, new_residual, indices
    
    def get_communication_bits(self, num_params: int) -> int:
        """Get number of bits for compressed message."""
        k = max(1, int(self.ratio * num_params))
        value_bits = k * 32
        index_bits = k * 32
        return value_bits + index_bits
    
    def get_communication_mb(self, num_params: int) -> float:
        """Get communication size in MB."""
        bits = self.get_communication_bits(num_params)
        return bits / (8 * 1024 * 1024)


class QSGDCompressor:
    """Quantized SGD compression."""
    
    def __init__(self, num_levels: int = 8):
        self.num_levels = num_levels
        self.bits_per_param = max(1, int(math.ceil(math.log2(num_levels))))
    
    def compress(self, update: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compress update using QSGD.
        
        Returns:
            quantized: quantized tensor
            signs: sign of each element
            norm: L2 norm of original update
        """
        norm = torch.norm(update)
        if norm < 1e-10:
            return torch.zeros_like(update), torch.ones_like(update), 0.0
        
        normalized = torch.abs(update) / norm
        
        scaled = normalized * (self.num_levels - 1)
        lower = torch.floor(scaled)
        prob = scaled - lower
        
        rand = torch.rand_like(prob)
        quantized_levels = torch.where(rand < prob, lower + 1, lower)
        
        signs = torch.sign(update)
        signs[signs == 0] = 1
        
        return quantized_levels, signs, norm.item()
    
    def decompress(
        self,
        quantized: torch.Tensor,
        signs: torch.Tensor,
        norm: float
    ) -> torch.Tensor:
        """Decompress QSGD quantized tensor."""
        if norm < 1e-10:
            return torch.zeros_like(quantized)
        
        normalized = quantized / (self.num_levels - 1)
        return signs * normalized * norm
    
    def get_communication_bits(self, num_params: int) -> int:
        """Get number of bits for compressed message."""
        quantized_bits = num_params * self.bits_per_param
        sign_bits = num_params
        norm_bits = 32
        return quantized_bits + sign_bits + norm_bits
    
    def get_communication_mb(self, num_params: int) -> float:
        """Get communication size in MB."""
        bits = self.get_communication_bits(num_params)
        return bits / (8 * 1024 * 1024)


def aggregate_compressed_topk(
    updates: list,
    weights: Optional[list] = None
) -> torch.Tensor:
    """Aggregate Top-K compressed updates."""
    if len(updates) == 0:
        raise ValueError("No updates to aggregate")
    
    if weights is None:
        weights = [1.0 / len(updates)] * len(updates)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]
    
    aggregated = torch.zeros_like(updates[0])
    for update, weight in zip(updates, weights):
        aggregated += weight * update
    
    return aggregated


def aggregate_qsgd(
    compressed_updates: list,
    weights: Optional[list] = None
) -> torch.Tensor:
    """Aggregate QSGD compressed updates after decompression."""
    if len(compressed_updates) == 0:
        raise ValueError("No updates to aggregate")
    
    compressor = QSGDCompressor()
    
    decompressed = []
    for quantized, signs, norm in compressed_updates:
        decompressed.append(compressor.decompress(quantized, signs, norm))
    
    return aggregate_compressed_topk(decompressed, weights)
