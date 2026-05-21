"""
topology.py — Dynamic communication graph for UAV-assisted HFL.

This module replaces the previous `clustering_coefficient = uniform[0.3, 1.0]`
random-scalar abuse with a real, defensible graph object.

================================================================================
Graph model
================================================================================
G_t = (V, E_t, w_t)  built each round from device 2-D positions:

  edge (i, j) ∈ E_t   iff   ‖p_i − p_j‖_2 ≤ r_comm   AND   both i, j active
  edge weight w_ij    = f(distance, bandwidth, channel_quality, reliability)

The graph evolves between rounds (bandwidth noise, transient dropouts, optional
UAV mobility).  This is the dynamic-topology piece from the brief.

================================================================================
Metrics — what each one means and why we use it
================================================================================

CHEAP, every round:
  • degree_centrality   : connectivity of node i — cluster-head candidates need
                          enough neighbours to actually receive updates.
  • local CC            : real graph-theoretic clustering coefficient; high CC
                          means neighbourhoods are dense (good for redundancy /
                          gossip robustness).
  • subgraph diameter   : worst-case hop count inside each cluster (used for
                          intra-cluster aggregation latency upper-bound).

EXPENSIVE, every K_topo rounds:
  • APL                  : average shortest-path length on largest CC of G_t.
                           Reported, not used in the per-round selection score.
  • betweenness          : how often a node sits on shortest paths — proxy for
                           "if this node fails, paths break".

================================================================================
Topology utility T_i (the one place T_i feeds into selection)
================================================================================

T_i is defined with a single, defensible claim:

    "Cluster-head candidates with high intra-cluster closeness reduce expected
     hop-count for intra-cluster aggregation."

So:   T_i  =  closeness_centrality_i (within its cluster subgraph)

NOT a soup of metrics — one metric, one claim.  The other metrics are reported
for analysis / plots, not used for selection.

================================================================================
References
================================================================================
  Watts & Strogatz 1998, "Collective dynamics of small-world networks"  (CC)
  Freeman 1977, "A set of measures of centrality based on betweenness"
  Newman 2010, "Networks: An Introduction"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Edge-weight model
# ─────────────────────────────────────────────────────────────────────────────

def edge_quality(dist: float, bw_i: float, bw_j: float, r_comm: float) -> float:
    """
    Symmetric edge quality in [0, 1].
      - distance term: 1 − d/r_comm (linearly decays inside the radius)
      - bandwidth term: harmonic mean of endpoint bandwidths, normalised
    Picking a *quality* rather than raw cost keeps NetworkX algorithms (which
    expect non-negative weights for shortest paths) well-behaved; we invert to
    cost only when computing path-length-style metrics.
    """
    dist_term = max(0.0, 1.0 - dist / r_comm)
    bw_harmonic = 2.0 * bw_i * bw_j / (bw_i + bw_j + 1e-9)
    bw_term = min(1.0, bw_harmonic / 10.0)         # bandwidth saturates at 10 Mbps
    return float(dist_term * bw_term)


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(devices, r_comm: float) -> nx.Graph:
    """
    Build the round-t communication graph from current device state.
    Devices with `is_active == False` contribute no edges.
    """
    G = nx.Graph()
    for d in devices:
        G.add_node(
            d.device_id,
            position=d.position,
            bandwidth=d.bandwidth,
            compute_power=d.compute_power,
            active=d.is_active,
        )

    active = [d for d in devices if d.is_active]
    for i, di in enumerate(active):
        for dj in active[i + 1:]:
            d_ij = float(np.linalg.norm(np.asarray(di.position) - np.asarray(dj.position)))
            if d_ij <= r_comm:
                w = edge_quality(d_ij, di.bandwidth, dj.bandwidth, r_comm)
                if w > 1e-3:                       # skip near-zero edges
                    G.add_edge(di.device_id, dj.device_id,
                               weight=w, distance=d_ij, cost=1.0 - w + 1e-3)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopologyMetrics:
    """Per-round graph metrics. Some fields populated only every K_topo rounds."""
    round: int
    n_nodes_active: int
    n_edges: int
    density: float
    largest_cc_size: int

    # cheap (per-round)
    degree_centrality: Dict[int, float]
    local_clustering:  Dict[int, float]            # real CC, per node
    avg_local_cc:      float                       # global avg of local CC

    # expensive (every K_topo rounds; cached otherwise)
    apl: Optional[float] = None
    betweenness: Optional[Dict[int, float]] = None

    def to_row(self) -> Dict:
        return {
            "round": self.round,
            "n_nodes_active": self.n_nodes_active,
            "n_edges": self.n_edges,
            "density": self.density,
            "largest_cc_size": self.largest_cc_size,
            "avg_local_cc": self.avg_local_cc,
            "apl": self.apl if self.apl is not None else float("nan"),
        }


def compute_metrics(G: nx.Graph, round_idx: int, *,
                    compute_expensive: bool = False) -> TopologyMetrics:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = (2.0 * e) / (n * (n - 1)) if n > 1 else 0.0

    if e == 0:
        empty = {nid: 0.0 for nid in G.nodes}
        return TopologyMetrics(
            round=round_idx, n_nodes_active=n, n_edges=0, density=0.0,
            largest_cc_size=1, degree_centrality=empty, local_clustering=empty,
            avg_local_cc=0.0, apl=None, betweenness=None,
        )

    deg = nx.degree_centrality(G)
    lcc = nx.clustering(G)                         # local CC per node (real)
    avg_cc = float(np.mean(list(lcc.values()))) if lcc else 0.0

    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    apl = None
    bet = None
    if compute_expensive and len(largest) >= 2:
        H = G.subgraph(largest)
        # APL on largest CC (using `cost` so it reflects link quality)
        apl = float(nx.average_shortest_path_length(H, weight="cost"))
        bet = nx.betweenness_centrality(H, weight="cost")
        # extend with zeros for nodes not in largest CC
        for nid in G.nodes:
            bet.setdefault(nid, 0.0)

    return TopologyMetrics(
        round=round_idx, n_nodes_active=n, n_edges=e, density=density,
        largest_cc_size=len(largest), degree_centrality=deg, local_clustering=lcc,
        avg_local_cc=avg_cc, apl=apl, betweenness=bet,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Topology utility T_i (cluster-aware closeness centrality)
# ─────────────────────────────────────────────────────────────────────────────

def topology_utility(G: nx.Graph, clusters: Dict[int, List[int]]) -> Dict[int, float]:
    """
    T_i = closeness centrality of node i within its cluster's subgraph.

    Single defensible claim: a cluster-head candidate with high intra-cluster
    closeness reduces the expected hop count for intra-cluster aggregation,
    which in turn lowers cluster-internal latency.

    Returns T_i ∈ [0, 1] per device. Nodes in singleton or empty clusters get 0.
    """
    T: Dict[int, float] = {nid: 0.0 for nid in G.nodes}

    for _, members in clusters.items():
        members_in_G = [m for m in members if m in G.nodes]
        if len(members_in_G) < 2:
            continue
        H = G.subgraph(members_in_G)
        if H.number_of_edges() == 0:
            continue
        # Closeness on largest connected component of the subgraph
        comps = list(nx.connected_components(H))
        largest = max(comps, key=len)
        Hcc = H.subgraph(largest)
        closeness = nx.closeness_centrality(Hcc, distance="cost")
        for nid, c in closeness.items():
            T[nid] = float(c)

    return T


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic evolution between rounds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopologyEvolutionConfig:
    bw_noise_std:   float = 0.10      # multiplicative std on bandwidth
    dropout_prob:   float = 0.02      # per-round per-node dropout
    mobility_step:  float = 0.0       # 0.0 = stationary; >0 = jitter in metres
    area_size:      float = 500.0     # keep positions inside [0, area_size]²
    bw_baseline_mb: Tuple[float, float] = (1.0, 10.0)  # clip bw to this range


def evolve_devices(devices, evo: TopologyEvolutionConfig, rng: np.random.RandomState):
    """
    Mutates device fields in-place to model dynamic network conditions.
      • bandwidth fluctuates around its baseline (log-normal-ish multiplicative noise)
      • each node may temporarily drop (is_active = False) for this round
      • optional Brownian-ish position jitter (mobility_step)
    """
    bw_lo, bw_hi = evo.bw_baseline_mb
    for d in devices:
        # bandwidth fluctuation (around the baseline, not compounding)
        eps = float(rng.normal(0.0, evo.bw_noise_std))
        d.bandwidth = float(np.clip(d.bandwidth_baseline * (1.0 + eps), bw_lo, bw_hi))

        # transient dropout
        d.is_active = bool(rng.uniform() > evo.dropout_prob)

        # mobility (optional)
        if evo.mobility_step > 0:
            dx, dy = rng.normal(0.0, evo.mobility_step, size=2)
            x, y = d.position
            d.position = (
                float(np.clip(x + dx, 0.0, evo.area_size)),
                float(np.clip(y + dy, 0.0, evo.area_size)),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: one-shot per-round update
# ─────────────────────────────────────────────────────────────────────────────

def step_topology(devices, evo: TopologyEvolutionConfig, r_comm: float,
                  round_idx: int, *, compute_expensive: bool,
                  rng: np.random.RandomState) -> Tuple[nx.Graph, TopologyMetrics]:
    """One-call wrapper: evolve devices, rebuild G, compute metrics."""
    evolve_devices(devices, evo, rng)
    G = build_graph(devices, r_comm)
    M = compute_metrics(G, round_idx, compute_expensive=compute_expensive)
    return G, M
