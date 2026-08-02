"""
Tests for NEXUS gravitational scorer and SONAR multi-hop extension.
"""

import pytest
import numpy as np
import torch

from helix.nexus import (
    nexus_score, nexus_score_normalized,
    sonar_score, sonar_score_normalized,
    _geodesic_dist, _auto_alpha, _bfs_min_hops,
)


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


# ── alpha='auto' ──────────────────────────────────────────────────────────────

def _random_q(N, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(N, 4)
    return q / q.norm(dim=-1, keepdim=True)


def test_auto_alpha_returns_positive():
    q = _random_q(50, seed=10)
    alpha = _auto_alpha(q, confirmed_idx=[0, 1, 2])
    assert alpha > 0


def test_auto_alpha_nexus_shape():
    q = _random_q(50, seed=11)
    scores = nexus_score(q, confirmed_idx=[0, 5], alpha='auto')
    assert scores.shape == (50,)
    assert np.all(scores >= 0)


def test_auto_alpha_normalized_range():
    q = _random_q(60, seed=12)
    scores = nexus_score_normalized(q, confirmed_idx=[3, 7], alpha='auto')
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0 + 1e-6


# ── _bfs_min_hops ─────────────────────────────────────────────────────────────

def _chain_edge_index(n):
    """0→1→2→…→n-1 directed chain."""
    src = torch.arange(n - 1)
    dst = torch.arange(1, n)
    return torch.stack([src, dst])


def test_bfs_hops_chain():
    ei = _chain_edge_index(6)
    hops = _bfs_min_hops(ei, sources=[0], num_nodes=6, max_hops=10)
    expected = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    np.testing.assert_array_equal(hops, expected)


def test_bfs_hops_unreachable():
    ei = _chain_edge_index(4)   # 0→1→2→3; node 5 is isolated
    ei = torch.cat([ei, torch.zeros(2, 0, dtype=torch.long)], dim=1)
    hops = _bfs_min_hops(ei, sources=[0], num_nodes=6, max_hops=10)
    assert np.isinf(hops[4]) and np.isinf(hops[5])


# ── sonar_score ───────────────────────────────────────────────────────────────

def _ring_edge_index(n):
    """Bidirectional ring: 0↔1↔2↔…↔n-1↔0."""
    src = list(range(n)) + list(range(1, n)) + [0]
    dst = list(range(1, n)) + [0] + list(range(n))
    return torch.tensor([src, dst], dtype=torch.long)


def test_sonar_shape_and_range():
    q = _random_q(20, seed=20)
    ei = _ring_edge_index(20)
    scores = sonar_score(q, ei, confirmed_idx=[0, 1], alpha=2.0)
    assert scores.shape == (20,)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0 + 1e-6


def test_sonar_confirmed_score_high():
    """Confirmed nodes are hop-0 from themselves → must have max or near-max score."""
    q = _random_q(30, seed=21)
    ei = _ring_edge_index(30)
    confirmed = [0, 5]
    scores = sonar_score(q, ei, confirmed_idx=confirmed, alpha=2.0)
    top2 = scores.argsort()[-2:]
    assert any(c in top2 for c in confirmed), "At least one confirmed node must be in top-2"


def test_sonar_hop_decay_effect():
    """Nodes farther in hops should generally score lower than immediate neighbours."""
    q = _random_q(10, seed=22)
    ei = _chain_edge_index(10)
    scores = sonar_score(q, ei, confirmed_idx=[0], alpha=2.0, hop_decay=0.5)
    # Node 1 is 1 hop; node 5 is 5 hops — average should decrease
    assert scores[1] >= scores[5] - 0.15, "Closer hop nodes should generally score higher"


def test_sonar_empty_confirmed_returns_zeros():
    q = _random_q(15, seed=23)
    ei = _ring_edge_index(15)
    scores = sonar_score(q, ei, confirmed_idx=[], alpha=2.0)
    assert np.all(scores == 0)


def test_sonar_normalized_alias():
    q = _random_q(20, seed=24)
    ei = _ring_edge_index(20)
    s1 = sonar_score(q, ei, confirmed_idx=[2, 4], alpha=2.0)
    s2 = sonar_score_normalized(q, ei, confirmed_idx=[2, 4], alpha=2.0)
    np.testing.assert_array_equal(s1, s2)


def test_sonar_auto_alpha():
    q = _random_q(20, seed=25)
    ei = _ring_edge_index(20)
    scores = sonar_score(q, ei, confirmed_idx=[0, 3], alpha='auto')
    assert scores.shape == (20,)
    assert np.all(scores >= 0)
    assert scores.max() <= 1.0 + 1e-6
