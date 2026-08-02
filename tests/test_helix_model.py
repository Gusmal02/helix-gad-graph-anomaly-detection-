"""
Tests for HelixModel — shapes, gradients, laser query, geometric properties.
All tests use synthetic graphs (no real datasets required).
"""

import pytest
import torch
import numpy as np

from helix.models.helix import HelixModel
from helix.core.laplacian import normalized_laplacian


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_graph(N=50, E=80, D=10, seed=42):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.standard_normal((N, D)), dtype=torch.float32)
    src = rng.integers(0, N, E)
    dst = rng.integers(0, N, E)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return x, edge_index


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_forward_output_shapes():
    N, D = 50, 10
    x, ei = make_graph(N=N, D=D)
    model = HelixModel(in_dim=D)
    logits, q_final = model(x, ei)
    assert logits.shape == (N, 1), f"Expected (N,1), got {logits.shape}"
    assert q_final.shape == (N, 4), f"Expected (N,4), got {q_final.shape}"


def test_quaternion_unit_norm():
    """q_final should stay on S³ — all norms ≈ 1."""
    x, ei = make_graph()
    model = HelixModel(in_dim=10)
    _, q = model(x, ei)
    norms = q.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Quaternion norms off: min={norms.min():.6f}, max={norms.max():.6f}"


def test_laser_query_shape():
    N, D = 50, 10
    x, ei = make_graph(N=N, D=D)
    model = HelixModel(in_dim=D)
    logits = model.laser_query(x, ei)
    assert logits.shape == (N, 1)


def test_laser_vs_full_auc_comparable():
    """Laser and full forward should produce correlated scores (not identical)."""
    from sklearn.metrics import roc_auc_score
    N, D = 100, 8
    rng = np.random.default_rng(0)
    x, ei = make_graph(N=N, D=D)
    model = HelixModel(in_dim=D)
    model.eval()
    with torch.no_grad():
        logits_full, _ = model(x, ei)
        logits_laser   = model.laser_query(x, ei)
    # Pearson correlation between scores should be positive
    s_full  = logits_full.squeeze().numpy()
    s_laser = logits_laser.squeeze().numpy()
    corr = np.corrcoef(s_full, s_laser)[0, 1]
    assert corr > 0.5, f"Laser and full scores diverged (corr={corr:.3f})"


# ── Gradient tests ────────────────────────────────────────────────────────────

def test_gradients_flow_through_rotor():
    """Loss on logits should produce non-zero gradients in rotor parameters."""
    x, ei = make_graph(N=30, D=6)
    model = HelixModel(in_dim=6)
    logits, _ = model(x, ei)
    loss = logits.mean()
    loss.backward()

    for name, p in model.rotor.named_parameters():
        assert p.grad is not None, f"No gradient on rotor.{name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient on rotor.{name}"


def test_no_nan_in_output():
    """No NaN anywhere in forward pass, even with degenerate edges."""
    x, ei = make_graph(N=20, D=5, E=5)  # very sparse
    model = HelixModel(in_dim=5)
    logits, q = model(x, ei)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isnan(q).any(), "NaN in q_final"


# ── Directed graph test ───────────────────────────────────────────────────────

def test_directed_graph_no_nan():
    """Directed graphs (asymmetric edges) must not produce NaN via Laplacian."""
    N, D = 40, 8
    rng = np.random.default_rng(7)
    x = torch.randn(N, D)
    # Directed: row→col only, no reverse edges
    src = rng.integers(0, N // 2, 60)   # only left→right edges
    dst = rng.integers(N // 2, N, 60)
    ei  = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    model = HelixModel(in_dim=D)
    logits, q = model(x, ei)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(q).any()


# ── Config variants ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("hidden,k,t", [
    (3,  4, 5),   # minimal (original GFCN config)
    (16, 4, 5),   # default library config
    (32, 2, 3),   # fast config
])
def test_model_variants(hidden, k, t):
    x, ei = make_graph(N=30, D=8)
    model = HelixModel(in_dim=8, hidden=hidden, cheb_k=k, t_steps=t)
    logits, q = model(x, ei)
    assert logits.shape == (30, 1)
    assert q.shape == (30, 4)
    assert not torch.isnan(logits).any()
