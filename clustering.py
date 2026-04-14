"""Graph-based clustering using clustering coefficient and average path length."""

import numpy as np
from typing import List, Dict, Tuple
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans
from devices import Device


def build_adjacency_matrix(devices: List[Device], d0: float = 50.0) -> np.ndarray:
    """Build weighted adjacency matrix based on distance."""
    n = len(devices)
    distances = np.array([d.properties.distance for d in devices])
    
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            weight = np.exp(-np.abs(distances[i] - distances[j]) / d0)
            adj[i, j] = weight
            adj[j, i] = weight
    
    return adj


def compute_clustering_coefficients(adj: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Compute local clustering coefficient for each node."""
    n = adj.shape[0]
    binary_adj = (adj > threshold).astype(float)
    cc = np.zeros(n)
    
    for i in range(n):
        neighbors = np.where(binary_adj[i] > 0)[0]
        k = len(neighbors)
        
        if k < 2:
            cc[i] = 0.0
            continue
        
        triangles = 0
        for ni in range(len(neighbors)):
            for nj in range(ni + 1, len(neighbors)):
                if binary_adj[neighbors[ni], neighbors[nj]] > 0:
                    triangles += 1
        
        possible_triangles = k * (k - 1) / 2
        cc[i] = triangles / possible_triangles if possible_triangles > 0 else 0.0
    
    return cc


def compute_average_path_lengths(adj: np.ndarray) -> np.ndarray:
    """Compute average shortest path length for each node using Dijkstra."""
    n = adj.shape[0]
    
    distance_matrix = np.where(adj > 0, 1.0 / adj, np.inf)
    np.fill_diagonal(distance_matrix, 0)
    
    shortest_paths = dijkstra(distance_matrix, directed=False)
    
    apl = np.zeros(n)
    for i in range(n):
        valid_paths = shortest_paths[i, :]
        valid_paths = valid_paths[valid_paths < np.inf]
        valid_paths = valid_paths[valid_paths > 0]
        apl[i] = np.mean(valid_paths) if len(valid_paths) > 0 else 0.0
    
    return apl


def normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val < 1e-10:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def cluster_devices(
    devices: List[Device],
    num_clusters: int,
    d0: float = 50.0,
    cc_weight: float = 0.5,
    compute_weight: float = 0.3,
    bandwidth_weight: float = 0.2,
    seed: int = 42
) -> Tuple[Dict[int, List[int]], Dict[int, int], np.ndarray, np.ndarray]:
    """
    Cluster devices using graph-based features.
    
    Returns:
        clusters: {cluster_id: [device_ids]}
        cluster_heads: {cluster_id: head_device_id}
        cc_values: clustering coefficients
        apl_values: average path lengths
    """
    adj = build_adjacency_matrix(devices, d0)
    
    cc = compute_clustering_coefficients(adj)
    apl = compute_average_path_lengths(adj)
    
    compute_powers = np.array([d.properties.compute_power for d in devices])
    
    features = np.column_stack([
        normalize(cc),
        normalize(apl),
        normalize(compute_powers)
    ])
    
    num_clusters = min(num_clusters, len(devices))
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(features)
    
    clusters = {i: [] for i in range(num_clusters)}
    for device_idx, cluster_id in enumerate(labels):
        clusters[cluster_id].append(device_idx)
    
    for cluster_id in list(clusters.keys()):
        if len(clusters[cluster_id]) == 0:
            largest_cluster = max(clusters.keys(), key=lambda x: len(clusters[x]))
            if len(clusters[largest_cluster]) > 1:
                moved_device = clusters[largest_cluster].pop()
                clusters[cluster_id].append(moved_device)
    
    cluster_heads = {}
    bandwidths = np.array([d.properties.bandwidth for d in devices])
    
    for cluster_id, device_ids in clusters.items():
        if len(device_ids) == 0:
            continue
        
        scores = []
        for device_id in device_ids:
            score = (
                cc_weight * cc[device_id] +
                compute_weight * normalize(compute_powers)[device_id] +
                bandwidth_weight * normalize(bandwidths)[device_id]
            )
            scores.append((device_id, score))
        
        head_id = max(scores, key=lambda x: x[1])[0]
        cluster_heads[cluster_id] = head_id
    
    return clusters, cluster_heads, cc, apl


def get_cluster_info(
    clusters: Dict[int, List[int]],
    cluster_heads: Dict[int, int],
    devices: List[Device]
) -> Dict:
    """Get summary information about clusters."""
    info = {
        "num_clusters": len(clusters),
        "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        "cluster_heads": cluster_heads,
        "head_properties": {}
    }
    
    for cluster_id, head_id in cluster_heads.items():
        device = devices[head_id]
        info["head_properties"][cluster_id] = {
            "compute_power": device.properties.compute_power,
            "bandwidth": device.properties.bandwidth,
            "distance": device.properties.distance
        }
    
    return info
