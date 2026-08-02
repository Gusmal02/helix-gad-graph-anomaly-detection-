"""
NEXUS — gravitational anomaly scorer.

Given a trained HELIX model and a set of confirmed anomalous nodes,
propagates a risk score through the S³ manifold without retraining.

Score formula:
  nexus(j) = Σ_{i ∈ confirmed} exp(-α · arccos(|⟨q_i, q_j⟩|))

SONAR extends NEXUS by combining S³ proximity with graph-hop distance:
  sonar(j) = nexus(j) · hop_decay^(min_hops_from_confirmed)

Nodes reachable in few hops AND geometrically close score highest.
"""

import torch
import numpy as np
from collections import deque
from typing import Sequence


def _geodesic_dist(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance on S³ between two sets of unit quaternions.
    q1: (M, 4), q2: (N, 4). Returns (M, N).
    """
    dot = (q1.unsqueeze(1) * q2.unsqueeze(0)).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(dot.abs())   # symmetric, range [0, π/2]


def _auto_alpha(q_final: torch.Tensor, confirmed_idx: Sequence[int]) -> float:
    """
    Calibrate alpha as 1 / median geodesic distance from confirmed seeds to all nodes.
    Makes NEXUS scale-invariant across domains where S³ spread varies.
    """
    with torch.no_grad():
        q = q_final.detach().cpu()
        seeds = q[list(confirmed_idx)]
        dist = _geodesic_dist(seeds, q)          # (M, N)
        median_dist = dist.median().item()
    return 1.0 / max(median_dist, 1e-6)


def _bfs_min_hops(
    edge_index: torch.Tensor,
    sources: Sequence[int],
    num_nodes: int,
    max_hops: int,
) -> np.ndarray:
    """
    BFS from all source nodes simultaneously.
    Returns min_hops array (N,) — inf for unreachable nodes.
    """
    min_hops = np.full(num_nodes, np.inf)
    for s in sources:
        min_hops[s] = 0.0

    # Build adjacency list once
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        adj[u].append(v)

    queue = deque((s, 0) for s in sources)
    while queue:
        node, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for nb in adj[node]:
            if min_hops[nb] > hops + 1:
                min_hops[nb] = hops + 1
                queue.append((nb, hops + 1))

    return min_hops


def nexus_score(
    q_final: torch.Tensor,
    confirmed_idx: Sequence[int],
    alpha: float | str = 2.0,
) -> np.ndarray:
    """
    Compute NEXUS gravitational scores for all nodes.

    Parameters
    ----------
    q_final       : (N, 4) quaternion embeddings from a trained HELIX model
    confirmed_idx : indices of nodes confirmed as anomalous (seed set)
    alpha         : decay rate, or 'auto' to calibrate from data

    Returns
    -------
    scores : (N,) numpy array in [0, len(confirmed_idx)]
    """
    if len(confirmed_idx) == 0:
        return np.zeros(q_final.shape[0])

    if alpha == 'auto':
        alpha = _auto_alpha(q_final, confirmed_idx)

    with torch.no_grad():
        q = q_final.detach().cpu()
        seeds = q[list(confirmed_idx)]
        dist  = _geodesic_dist(seeds, q)
        scores = torch.exp(-alpha * dist).sum(dim=0)

    return scores.numpy()


def nexus_score_normalized(
    q_final: torch.Tensor,
    confirmed_idx: Sequence[int],
    alpha: float | str = 2.0,
) -> np.ndarray:
    """Same as nexus_score but normalized to [0, 1]."""
    raw = nexus_score(q_final, confirmed_idx, alpha)
    denom = max(len(confirmed_idx), 1)
    return raw / denom


def sonar_score(
    q_final: torch.Tensor,
    edge_index: torch.Tensor,
    confirmed_idx: Sequence[int],
    alpha: float | str = 2.0,
    max_hops: int = 4,
    hop_decay: float = 0.6,
) -> np.ndarray:
    """
    SONAR: combines S³ geodesic proximity (NEXUS) with graph-hop distance.

    sonar(j) = nexus(j) · hop_decay^(min_hops_to_confirmed)

    Nodes that are BOTH geometrically close in S³ AND reachable in few hops
    from confirmed seeds receive the highest scores.
    Unreachable nodes (> max_hops) still receive their NEXUS score × hop_decay^max_hops.

    Parameters
    ----------
    q_final       : (N, 4) quaternion embeddings
    edge_index    : (2, E) graph connectivity
    confirmed_idx : seed set of confirmed anomalous nodes
    alpha         : NEXUS decay rate, or 'auto'
    max_hops      : BFS depth limit
    hop_decay     : multiplicative penalty per hop (0 < hop_decay ≤ 1)

    Returns
    -------
    scores : (N,) numpy array in [0, 1]
    """
    if len(confirmed_idx) == 0:
        return np.zeros(q_final.shape[0])

    N = q_final.shape[0]
    nx = nexus_score_normalized(q_final, confirmed_idx, alpha)

    min_hops = _bfs_min_hops(edge_index, confirmed_idx, N, max_hops)
    # Cap unreachable nodes at max_hops for the decay factor
    min_hops = np.minimum(min_hops, max_hops)

    hop_weights = hop_decay ** min_hops
    combined = nx * hop_weights

    denom = combined.max()
    return combined / max(denom, 1e-9)


def sonar_score_normalized(
    q_final: torch.Tensor,
    edge_index: torch.Tensor,
    confirmed_idx: Sequence[int],
    alpha: float | str = 2.0,
    max_hops: int = 4,
    hop_decay: float = 0.6,
) -> np.ndarray:
    """Alias for sonar_score (always returns [0, 1])."""
    return sonar_score(q_final, edge_index, confirmed_idx, alpha, max_hops, hop_decay)
