"""
Tests for NEXUS gravitational scorer.
"""

import pytest
import numpy as np
import torch

from helix.nexus import nexus_score, nexus_score_normalized, _geodesic_dist


# ── Geodesic distance ─────────────────────────────────────────────────────────

def test_geodesic_identity_is_near_zero():
    """Distance from identity to itself should be near 0 (bounded by clamp precision)."""
    q = torch.zeros(5, 4)
    q[:, 0] = 1.0  # identity quaternion
    dist = _geodesic_dist(q, q)  # (5, 5)
    # clamp at 1-1e-7 → acos(1-1e-7) ≈ 4.5e-4; tolerance reflects this
    assert dist.max().item() < 1e-3, f"Expected near-zero, got {dist.max().item():.6f}"


def test_geodesic_antipodal_is_max():
    """Antipodal quaternions (q, -q) represent same rotation → dist = 0 via abs()."""
    q1 = torch.zeros(1, 4); q1[0, 0] = 1.0
    q2 = torch.zeros(1, 4); q2[0, 0] = -1.0  # -identity (same rotation)
    dist = _geodesic_dist(q1, q2)
    assert dist.item() < 1e-3


def test_geodesic_range():
    """All distances should be in [0, π/2]."""
    torch.manual_seed(0)
    q = torch.randn(20, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    dist = _geodesic_dist(q, q)
    assert (dist >= 0).all()
    assert (dist <= torch.pi / 2 + 1e-5).all()


# ── nexus_score ───────────────────────────────────────────────────────────────

def test_score_shape():
    torch.manual_seed(1)
    q = torch.randn(50, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    scores = nexus_score(q, confirmed_idx=[0, 1, 2], alpha=2.0)
    assert scores.shape == (50,)


def test_confirmed_nodes_score_high():
    """Confirmed nodes should have the highest NEXUS scores."""
    torch.manual_seed(2)
    N = 100
    q = torch.randn(N, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    confirmed = [0, 5, 10]
    scores = nexus_score(q, confirmed_idx=confirmed, alpha=3.0)
    for idx in confirmed:
        # Confirmed nodes are distance 0 from themselves → max contribution
        others = [i for i in range(N) if i not in confirmed]
        assert scores[idx] >= scores[others].max() - 1e-3, \
            f"Confirmed node {idx} doesn't have max score"


def test_empty_confirmed_returns_zeros():
    torch.manual_seed(3)
    q = torch.randn(30, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    scores = nexus_score(q, confirmed_idx=[], alpha=2.0)
    assert (scores == 0).all()


def test_normalized_range():
    torch.manual_seed(4)
    q = torch.randn(50, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    scores = nexus_score_normalized(q, confirmed_idx=[0, 1, 2], alpha=2.0)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0 + 1e-6


def test_alpha_decay():
    """Higher alpha → scores fall off faster → more concentrated."""
    torch.manual_seed(5)
    q = torch.randn(40, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    confirmed = [0]
    s_low  = nexus_score(q, confirmed, alpha=0.5)
    s_high = nexus_score(q, confirmed, alpha=5.0)
    # With high alpha, variance across nodes should be larger (more concentrated)
    assert s_high.std() > s_low.std(), "High alpha should produce more concentrated scores"
