"""
NEXUS — gravitational anomaly scorer.

Given a trained HELIX model and a set of confirmed anomalous nodes,
propagates a risk score through the S³ manifold without retraining.

Score formula:
  nexus(j) = Σ_{i ∈ confirmed} exp(-α · arccos(|⟨q_i, q_j⟩|))

Interpretation: nodes close to confirmed anomalies in quaternion space
receive high scores regardless of their label. Useful for:
  - Semi-supervised expansion from a few known fraud cases
  - Re-scoring after new confirmations without full retraining
"""

import torch
import numpy as np
from typing import Sequence


def _geodesic_dist(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance on S³ between two sets of unit quaternions.
    q1: (M, 4), q2: (N, 4). Returns (M, N).
    """
    dot = (q1.unsqueeze(1) * q2.unsqueeze(0)).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(dot.abs())   # symmetric, range [0, π/2]


def nexus_score(
    q_final: torch.Tensor,
    confirmed_idx: Sequence[int],
    alpha: float = 2.0,
) -> np.ndarray:
    """
    Compute NEXUS gravitational scores for all nodes.

    Parameters
    ----------
    q_final       : (N, 4) quaternion embeddings from a trained HELIX model
    confirmed_idx : indices of nodes confirmed as anomalous (seed set)
    alpha         : decay rate — higher alpha → scores fall off faster with distance

    Returns
    -------
    scores : (N,) numpy array in [0, len(confirmed_idx)]
             Normalize by len(confirmed_idx) for [0, 1] range.
    """
    if len(confirmed_idx) == 0:
        return np.zeros(q_final.shape[0])

    with torch.no_grad():
        q = q_final.detach().cpu()
        seeds = q[list(confirmed_idx)]        # (M, 4)
        dist  = _geodesic_dist(seeds, q)      # (M, N)
        scores = torch.exp(-alpha * dist).sum(dim=0)  # (N,)

    return scores.numpy()


def nexus_score_normalized(
    q_final: torch.Tensor,
    confirmed_idx: Sequence[int],
    alpha: float = 2.0,
) -> np.ndarray:
    """Same as nexus_score but normalized to [0, 1]."""
    raw = nexus_score(q_final, confirmed_idx, alpha)
    denom = max(len(confirmed_idx), 1)
    return raw / denom
