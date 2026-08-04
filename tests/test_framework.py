"""
End-to-end tests for HelixFramework.

Fast tests use tiny synthetic graphs (N=80, E=100, D=6, epochs=5).
Slow tests (marked @pytest.mark.slow) run full 200-epoch training.
"""

import pytest
import warnings
import numpy as np
import torch

from helix.framework import HelixFramework, ExplainResult
from helix.trainer import TrainConfig


# ── Synthetic dataset fixture ─────────────────────────────────────────────────

def make_dataset(N=80, D=6, E=100, fraud_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, D)).astype(np.float32)
    src = rng.integers(0, N, E)
    dst = rng.integers(0, N, E)
    ei  = np.stack([src, dst]).astype(np.int64)
    labels = rng.choice([0, 1], size=N, p=[1 - fraud_rate, fraud_rate]).astype(np.float32)
    return x, ei, labels


_FAST_CFG = TrainConfig(epochs=5, verbose=False)


# ── fit + predict ─────────────────────────────────────────────────────────────

def test_fit_predict_shapes():
    x, ei, y = make_dataset()
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG)
    scores = fw.predict(x, ei)
    assert scores.shape == (x.shape[0],)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_predict_before_fit_raises():
    fw = HelixFramework()
    x, ei, y = make_dataset()
    with pytest.raises(RuntimeError, match="fit"):
        fw.predict(x, ei)


@pytest.mark.parametrize("model_name", ["HELIX", "SAGE", "GCN", "MLP"])
def test_all_models_fit_and_predict(model_name):
    x, ei, y = make_dataset()
    fw = HelixFramework()
    fw.fit(x, ei, y, model=model_name, cfg=_FAST_CFG)
    scores = fw.predict(x, ei)
    assert scores.shape == (x.shape[0],)
    assert not np.isnan(scores).any()


def test_invalid_model_raises():
    x, ei, y = make_dataset()
    fw = HelixFramework()
    with pytest.raises(ValueError):
        fw.fit(x, ei, y, model="XGBoost", cfg=_FAST_CFG)


# ── explain ───────────────────────────────────────────────────────────────────

def test_explain_shapes():
    N, D = 80, 6
    x, ei, y = make_dataset(N=N, D=D)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG)
    result = fw.explain(x, ei)

    assert isinstance(result, ExplainResult)
    assert result.scores.shape == (N,)
    assert result.rho.shape == (N,)
    assert result.eta.shape == (N,)
    assert result.geo_dist.shape == (N,)
    assert result.torque.shape == (N,)
    assert result.q_final.shape == (N, 4)
    assert result.model_used == "HELIX"


def test_explain_with_sage_still_returns_geometry():
    """When SAGE is the best model, explain() still returns HELIX geometry."""
    x, ei, y = make_dataset()
    fw = HelixFramework()
    fw.fit(x, ei, y, model="SAGE", cfg=_FAST_CFG)
    result = fw.explain(x, ei)
    assert result.q_final is not None
    assert result.q_final.shape[1] == 4
    assert result.model_used == "SAGE"


def test_explain_warns_on_dense_graph():
    """Framework should warn when E/N > 10 and explain() is called."""
    N, D = 50, 6
    E = 600  # E/N = 12 > 10
    rng = np.random.default_rng(9)
    x = rng.standard_normal((N, D)).astype(np.float32)
    src = rng.integers(0, N, E)
    dst = rng.integers(0, N, E)
    ei  = np.stack([src, dst]).astype(np.int64)
    y   = rng.choice([0, 1], N, p=[0.9, 0.1]).astype(np.float32)

    fw = HelixFramework()
    fw.fit(x, ei, y, model="SAGE", cfg=_FAST_CFG)

    # Manually inject a mock validation report with high density
    from helix.validator import ValidationReport
    fw._validation_report = ValidationReport(
        recommended_model="SAGE", confidence="ALTA",
        reason="test", sigma_seeds=0.01,
        graph_density=12.0, gini=0.3, homophily=0.5,
        helix_auc_mean=0.8, helix_auc_std=0.01,
        mlp_auc=0.75, lift=0.05,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fw.explain(x, ei)
        assert any("dense" in str(warning.message).lower() for warning in w), \
            "Expected UserWarning about dense graph"


# ── nexus ─────────────────────────────────────────────────────────────────────

def test_nexus_shapes_and_range():
    N = 80
    x, ei, y = make_dataset(N=N)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG)
    confirmed = [0, 5, 10]
    scores = fw.nexus(x, ei, confirmed=confirmed)
    assert scores.shape == (N,)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0 + 1e-6


def test_nexus_confirmed_highest():
    """Confirmed anomaly nodes should rank in top scores."""
    N = 100
    x, ei, y = make_dataset(N=N)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG)
    confirmed = [2, 7]
    scores = fw.nexus(x, ei, confirmed=confirmed)
    top10 = set(np.argsort(scores)[-10:])
    assert any(c in top10 for c in confirmed), \
        "Confirmed nodes should appear in top-10 NEXUS scores"


# ── validate (standalone) ─────────────────────────────────────────────────────

@pytest.mark.slow
def test_validate_returns_report():
    x, ei, y = make_dataset(N=200, D=8, E=300)
    fw = HelixFramework()
    report = fw.validate(x, ei, y.astype(np.int32))
    assert report.recommended_model in ("HELIX", "SAGE", "GCN", "MLP")
    assert 0.0 <= report.helix_auc_mean <= 1.0


# ── fit chaining ──────────────────────────────────────────────────────────────

def test_fit_returns_self():
    x, ei, y = make_dataset()
    fw = HelixFramework()
    result = fw.fit(x, ei, y, model="MLP", cfg=_FAST_CFG)
    assert result is fw


# ── sonar framework method ───────────────────────────────────────────────────

def test_sonar_shapes_and_range():
    N = 80
    x, ei, y = make_dataset(N=N)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG)
    scores = fw.sonar(x, ei, confirmed=[0, 5, 10])
    assert scores.shape == (N,)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0 + 1e-6


def test_sonar_auto_alpha():
    x, ei, y = make_dataset()
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG)
    scores = fw.sonar(x, ei, confirmed=[1, 3], alpha='auto')
    assert not np.isnan(scores).any()


# ── save / load ──────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    x, ei, y = make_dataset()
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG)
    scores_before = fw.predict(x, ei)

    path = tmp_path / "model.pt"
    fw.save(path)
    fw2 = HelixFramework.load(path)
    scores_after = fw2.predict(x, ei)

    np.testing.assert_allclose(scores_before, scores_after, atol=1e-2)


def test_save_load_sage(tmp_path):
    x, ei, y = make_dataset()
    fw = HelixFramework()
    fw.fit(x, ei, y, model="SAGE", cfg=_FAST_CFG)
    scores_before = fw.predict(x, ei)

    path = tmp_path / "sage.pt"
    fw.save(path)
    fw2 = HelixFramework.load(path)
    scores_after = fw2.predict(x, ei)

    np.testing.assert_allclose(scores_before, scores_after, atol=1e-6)
    assert fw2._best_model_name == "SAGE"


# ── auto PCA ─────────────────────────────────────────────────────────────────

def test_auto_pca():
    x, ei, y = make_dataset(D=20)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG, auto_pca=5)
    assert fw._pca is True
    scores = fw.predict(x, ei)
    assert scores.shape == (x.shape[0],)
    assert not np.isnan(scores).any()


def test_auto_pca_save_load(tmp_path):
    x, ei, y = make_dataset(D=20)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG, auto_pca=5)
    scores_before = fw.predict(x, ei)

    path = tmp_path / "pca.pt"
    fw.save(path)
    fw2 = HelixFramework.load(path)
    scores_after = fw2.predict(x, ei)

    np.testing.assert_allclose(scores_before, scores_after, atol=1e-2)


# ── early stopping ───────────────────────────────────────────────────────────

def test_early_stopping():
    x, ei, y = make_dataset()
    cfg = TrainConfig(epochs=200, patience=3)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=cfg)
    scores = fw.predict(x, ei)
    assert not np.isnan(scores).any()


# ── directed ─────────────────────────────────────────────────────────────────

def test_directed_framework():
    x, ei, y = make_dataset()
    fw = HelixFramework(directed=True)
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG)
    scores = fw.predict(x, ei)
    assert scores.shape == (x.shape[0],)
    assert not np.isnan(scores).any()


# ── torque in explain ────────────────────────────────────────────────────────

def test_torque_in_explain():
    x, ei, y = make_dataset()
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=_FAST_CFG)
    result = fw.explain(x, ei)
    assert result.torque.shape == (x.shape[0],)
    assert (result.torque >= 0).all()


# ── torque lambda ────────────────────────────────────────────────────────────

def test_torque_lambda():
    x, ei, y = make_dataset()
    cfg = TrainConfig(epochs=5, torque_lambda=0.1)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=cfg)
    scores = fw.predict(x, ei)
    assert not np.isnan(scores).any()


# ── centroid repulsion ───────────────────────────────────────────────────────

def test_centroid_lambda():
    x, ei, y = make_dataset()
    cfg = TrainConfig(epochs=5, centroid_lambda=0.1)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="HELIX", cfg=cfg)
    scores = fw.predict(x, ei)
    assert not np.isnan(scores).any()


def test_centroid_lambda_sage_noop():
    """centroid_lambda silently ignored for non-HELIX models (no q_final)."""
    x, ei, y = make_dataset()
    cfg = TrainConfig(epochs=5, centroid_lambda=0.5)
    fw = HelixFramework()
    fw.fit(x, ei, y, model="SAGE", cfg=cfg)
    scores = fw.predict(x, ei)
    assert not np.isnan(scores).any()
