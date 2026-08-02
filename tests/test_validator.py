"""
Tests for Graph Validator — structural stats, decision rules, smoke test.

Full probe (100 epochs × 3 seeds) is skipped in CI (marked slow).
Decision rule logic is tested directly without running the probe.
"""

import pytest
import numpy as np
import torch

from helix.core.graph import edge_density, degree_gini, label_homophily, graph_stats
from helix.validator import _decide, ValidationReport


# ── Core graph stat functions ─────────────────────────────────────────────────

def test_edge_density_basic():
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    assert edge_density(ei, num_nodes=3) == pytest.approx(1.0)


def test_edge_density_sparse():
    N, E = 1000, 50
    rng = np.random.default_rng(0)
    src = rng.integers(0, N, E)
    dst = rng.integers(0, N, E)
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    assert edge_density(ei, N) == pytest.approx(E / N)


def test_degree_gini_uniform():
    """Regular graph → Gini near 0."""
    N = 20
    # Each node has exactly degree 2 (ring graph)
    src = list(range(N))
    dst = [(i + 1) % N for i in range(N)]
    ei = torch.tensor([src, dst], dtype=torch.long)
    gini = degree_gini(ei, N)
    assert gini < 0.2, f"Ring graph should have low Gini, got {gini:.3f}"


def test_degree_gini_hub():
    """Star graph (one hub) → Gini near 1."""
    N = 50
    src = [0] * (N - 1)
    dst = list(range(1, N))
    ei = torch.tensor([src, dst], dtype=torch.long)
    gini = degree_gini(ei, N)
    assert gini > 0.7, f"Star graph should have high Gini, got {gini:.3f}"


def test_label_homophily_perfect():
    """Edges only between same-class nodes → homophily = 1."""
    # 10 positive nodes, 10 negative; edges only within class
    ei = torch.tensor([[0, 1, 10, 11], [1, 2, 11, 12]], dtype=torch.long)
    labels = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    h = label_homophily(ei, labels)
    assert h == pytest.approx(1.0)


def test_label_homophily_zero():
    """Edges only between different classes → homophily = 0."""
    # 0=fraud, 1=legit, edges only cross-class
    ei = torch.tensor([[0, 0, 0], [3, 4, 5]], dtype=torch.long)
    labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long)
    h = label_homophily(ei, labels)
    assert h == pytest.approx(0.0)


# ── Decision rule tests ───────────────────────────────────────────────────────

def _stats(density, gini, homophily):
    return {"density": density, "gini": gini, "homophily": homophily}


def test_rule1_dense_graph():
    model, conf, reason = _decide(_stats(15.0, 0.3, 0.5), sigma=0.005, lift=0.05)
    assert model == "SAGE"
    assert conf == "ALTA"
    assert "dense" in reason.lower() or "10" in reason


def test_rule2_hub_and_spoke():
    model, conf, reason = _decide(_stats(2.0, 0.75, 0.97), sigma=0.005, lift=0.05)
    assert model == "SAGE"
    assert conf == "ALTA"


def test_rule2_not_triggered_low_homophily():
    """High Gini but low homophily → not hub-and-spoke → fall through to rule 3/4."""
    model, _, _ = _decide(_stats(2.0, 0.75, 0.60), sigma=0.010, lift=0.05)
    assert model == "HELIX"


def test_rule3_unstable_mlp():
    model, conf, _ = _decide(_stats(3.0, 0.3, 0.5), sigma=0.030, lift=-0.10)
    assert model == "MLP"
    assert conf == "ALTA"


def test_rule3_unstable_sage():
    model, conf, _ = _decide(_stats(3.0, 0.3, 0.5), sigma=0.030, lift=0.02)
    assert model == "SAGE"
    assert conf == "MEDIA"


def test_rule4_helix_high_confidence():
    model, conf, _ = _decide(_stats(2.0, 0.3, 0.5), sigma=0.010, lift=0.05)
    assert model == "HELIX"
    assert conf == "ALTA"


def test_rule4_helix_medium_confidence():
    model, conf, _ = _decide(_stats(2.0, 0.3, 0.5), sigma=0.020, lift=0.03)
    assert model == "HELIX"
    assert conf == "MEDIA"


# ── Smoke test: full probe on tiny synthetic graph ────────────────────────────

@pytest.mark.slow
def test_validate_smoke():
    """End-to-end smoke test with tiny synthetic data. Skipped unless -m slow."""
    from helix.validator import validate
    rng = np.random.default_rng(42)
    N, D, E = 200, 8, 300
    x = rng.standard_normal((N, D)).astype(np.float32)
    src = rng.integers(0, N, E)
    dst = rng.integers(0, N, E)
    ei  = np.stack([src, dst])
    labels = rng.choice([0, 1], size=N, p=[0.9, 0.1]).astype(np.int32)

    report = validate(x, ei, labels, N)

    assert report.recommended_model in ("HELIX", "SAGE", "GCN", "MLP")
    assert report.confidence in ("ALTA", "MEDIA")
    assert 0.0 <= report.helix_auc_mean <= 1.0
    assert report.sigma_seeds >= 0.0
    assert report.graph_density == pytest.approx(E / N)
