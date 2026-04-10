### clustering.py
import numpy as np
from sklearn.cluster import KMeans


def cluster_devices(devices, num_clusters, seed=42):
    """
    Cluster devices based on their spatial/network features.
    Returns list of lists (cluster_id -> list of device_ids).
    """
    features = np.array([
        [d.distance, d.bandwidth, d.compute_power, d.clustering_coefficient]
        for d in devices
    ])

    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features_norm = (features - mean) / std

    n_clusters = min(num_clusters, len(devices))
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(features_norm)

    clusters = [[] for _ in range(n_clusters)]
    for device_id, cluster_id in enumerate(labels):
        clusters[cluster_id].append(device_id)

    # Ensure no empty clusters
    non_empty = [c for c in clusters if len(c) > 0]
    return non_empty


def get_cluster_head(cluster_device_ids, devices):
    """Select cluster head: device with highest compute_power * bandwidth."""
    scores = [devices[d].compute_power * devices[d].bandwidth
              for d in cluster_device_ids]
    best_idx = np.argmax(scores)
    return cluster_device_ids[best_idx]
