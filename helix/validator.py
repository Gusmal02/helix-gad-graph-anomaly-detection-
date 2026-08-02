"""
Graph Validator — automatically selects HELIX, SAGE, or MLP for a given graph.

Algorithm:
  1. Compute structural stats: E/N density, degree Gini, edge homophily
  2. Quick probe: HELIX × 3 seeds × 100 epochs → σ_seeds + lift vs MLP
  3. Apply 4-rule decision function (calibrated on 4 financial fraud domains)

Empirical accuracy: 4/4 known domains (Elliptic, AMLSim, PaySim, Electricity).

Decision rules (in priority order):
  1. E/N > 10                          → SAGE (dense graph, HELIX over-smooths)
  2. Gini > 0.6 AND homophily > 0.95   → SAGE (hub-and-spoke community structure)
  3. σ_seeds >= 0.025                  → MLP (lift < -0.05) or SAGE otherwise
  4. σ_seeds < 0.025 AND E/N <= 10     → HELIX (sparse causal real graph)
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from helix.core.laplacian import normalized_laplacian
from helix.core.chebyshev import ChebyshevFNO
from helix.core.rotors import quat_exp, quat_mul, quat_normalize, quat_rotate
from helix.core.graph import graph_stats
from helix.metrics import sigma_seeds as compute_sigma

logger = logging.getLogger(__name__)

# Calibrated thresholds (100 epochs × 3 seeds)
_SIGMA_HIGH = 0.015
_SIGMA_MED  = 0.025
_DENSITY_DENSE = 10.0
_GINI_HUB      = 0.60
_HOM_HUB       = 0.95

_PROBE_EPOCHS = 100
_PROBE_SEEDS  = [42, 123, 777]
_PROBE_LR     = 1e-3


@dataclass
class ValidationReport:
    recommended_model: str       # 'HELIX', 'SAGE', or 'MLP'
    confidence: str              # 'ALTA' or 'MEDIA'
    reason: str
    sigma_seeds: float
    graph_density: float         # E/N
    gini: float
    homophily: float | None
    helix_auc_mean: float
    helix_auc_std: float
    mlp_auc: float
    lift: float                  # HELIX AUC − MLP AUC


# ── Minimal probe models (faster than full HELIX/MLP) ──────────────────────────

class _QuickHELIX(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.emit = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 3))
        self.fno  = ChebyshevFNO(order=4, num_channels=3)
        self.head = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        v = self.emit(x)
        Phi = self.fno(v, L_tilde)
        q = torch.zeros(N, 4, device=x.device); q[:, 0] = 1.0
        for _ in range(5):
            tau = torch.cross(v, Phi, dim=-1)
            dq  = quat_exp(tau * 0.05)
            q   = quat_normalize(quat_mul(dq, q))
            v   = quat_rotate(q, v)
        return self.head(v).squeeze(-1)


class _QuickMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),    nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _quick_train(
    ModelClass, x: torch.Tensor, L_arg,
    tr_idx, va_idx, y_tr, y_va,
    pos_w: float, seed: int,
) -> float:
    torch.manual_seed(seed)
    model = ModelClass(x.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=_PROBE_LR)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    for _ in range(_PROBE_EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(x, L_arg)
        loss   = crit(logits[tr_idx], y_tr_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = model(x, L_arg)[va_idx].numpy()
    return float(roc_auc_score(y_va, scores))


# ── Decision rule ───────────────────────────────────────────────────────────────

def _decide(stats: dict, sigma: float, lift: float) -> tuple[str, str, str]:
    density = stats["density"]
    gini    = stats["gini"]
    hom     = stats["homophily"] or 0.0

    if density > _DENSITY_DENSE:
        return "SAGE", "ALTA", f"E/N={density:.1f} > 10 (dense graph, HELIX over-smooths)"

    if gini > _GINI_HUB and hom > _HOM_HUB:
        return "SAGE", "ALTA", f"Gini={gini:.3f} + homophily={hom:.3f} (hub-and-spoke)"

    if sigma >= _SIGMA_MED:
        if lift < -0.05:
            return "MLP", "ALTA", f"σ={sigma:.4f} >= 0.025, lift={lift:+.3f} (unstable, MLP wins)"
        return "SAGE", "MEDIA", f"σ={sigma:.4f} >= 0.025, lift={lift:+.3f} (unstable)"

    conf = "ALTA" if sigma < _SIGMA_HIGH else "MEDIA"
    return "HELIX", conf, f"σ={sigma:.4f} < 0.025, E/N={density:.1f} <= 10 (sparse causal graph)"


# ── Public API ──────────────────────────────────────────────────────────────────

def validate(
    x: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
    num_nodes: int | None = None,
) -> ValidationReport:
    """
    Run the Graph Validator on a dataset.

    Parameters
    ----------
    x          : (N, D) node features as numpy array
    edge_index : (2, E) graph edges as numpy int array
    labels     : (N,) binary labels (0=normal, 1=anomalous); -1 = unlabeled
    num_nodes  : optional; inferred from labels if None

    Returns
    -------
    ValidationReport with recommendation and diagnostics
    """
    if num_nodes is None:
        num_nodes = int(x.shape[0])

    N  = num_nodes
    ei = torch.tensor(edge_index, dtype=torch.long)
    ew = torch.ones(ei.shape[1])
    X  = torch.tensor(x, dtype=torch.float32)

    stats = graph_stats(ei, N, torch.tensor(labels))

    # Stratified 70/30 split on labeled nodes
    known = np.where(labels >= 0)[0]
    pos   = known[labels[known] == 1]
    neg   = known[labels[known] == 0]
    rng   = np.random.default_rng(42)
    rng.shuffle(pos); rng.shuffle(neg)
    tr_idx = np.concatenate([pos[:int(0.7*len(pos))], neg[:int(0.7*len(neg))]])
    va_idx = np.concatenate([pos[int(0.7*len(pos)):],  neg[int(0.7*len(neg)):]])
    y_tr   = labels[tr_idx].astype(np.float32)
    y_va   = labels[va_idx]
    pos_w  = float(len(neg)) / max(float(len(pos)), 1.0)

    L_tilde, _ = normalized_laplacian(ei, ew, N)

    logger.info("Graph Validator: running HELIX probe (%d epochs × %d seeds)…",
                _PROBE_EPOCHS, len(_PROBE_SEEDS))

    helix_aucs = [
        _quick_train(_QuickHELIX, X, L_tilde, tr_idx, va_idx, y_tr, y_va, pos_w, s)
        for s in _PROBE_SEEDS
    ]
    mlp_auc = _quick_train(_QuickMLP, X, None, tr_idx, va_idx, y_tr, y_va, pos_w, 42)

    sigma  = compute_sigma(helix_aucs)
    mean_h = float(np.mean(helix_aucs))
    lift   = mean_h - mlp_auc

    model, conf, reason = _decide(stats, sigma, lift)

    logger.info("Graph Validator → %s (%s): %s", model, conf, reason)

    return ValidationReport(
        recommended_model=model,
        confidence=conf,
        reason=reason,
        sigma_seeds=sigma,
        graph_density=stats["density"],
        gini=stats["gini"],
        homophily=stats["homophily"],
        helix_auc_mean=mean_h,
        helix_auc_std=sigma,
        mlp_auc=mlp_auc,
        lift=lift,
    )
