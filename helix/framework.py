"""
HelixFramework — public API for the Helix anomaly detection library.

Usage:
    from helix import HelixFramework
    import numpy as np

    fw = HelixFramework()
    fw.fit(X, edge_index, labels)

    scores          = fw.predict(X, edge_index)          # fast (Laser Query)
    result          = fw.explain(X, edge_index)          # full loop + geometry
    nexus_scores    = fw.nexus(X, edge_index, confirmed=[12, 45])
    report          = fw.validate(X, edge_index, labels)
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from helix.models.helix import HelixModel
from helix.models.sage  import SAGEModel
from helix.models.gcn   import GCNModel
from helix.models.mlp   import MLPModel
from helix.trainer import TrainConfig, TrainResult, train, stratified_split
from helix.validator import validate, ValidationReport
from helix.nexus import nexus_score_normalized
from helix.metrics import rho, eta, geo_distance, best_f1_geo
from helix.core.laplacian import normalized_laplacian

logger = logging.getLogger(__name__)

_MODEL_CLASSES = {
    "HELIX": HelixModel,
    "SAGE":  SAGEModel,
    "GCN":   GCNModel,
    "MLP":   MLPModel,
}


@dataclass
class ExplainResult:
    scores: np.ndarray          # (N,) predicted anomaly probability
    rho: np.ndarray             # (N,) imaginary norm — instability
    eta: np.ndarray             # (N,) scalar part — identity alignment
    geo_dist: np.ndarray        # (N,) geodesic distance from identity
    model_used: str             # which model produced the scores
    confidence: str | None      # validator confidence if auto-selected
    q_final: np.ndarray | None  # (N, 4) quaternions — None for non-HELIX models


class HelixFramework:
    """
    Intelligent graph anomaly detection framework.

    Automatically selects the best model (HELIX, SAGE, GCN, or MLP)
    based on the structural properties of the input graph.

    Parameters
    ----------
    helix_hidden : int   Hidden dim of HELIX emission MLP (default 16)
    helix_k      : int   Chebyshev order (default 4)
    helix_t      : int   Rotor steps (default 5)
    gnn_hidden   : int   Hidden dim for SAGE/GCN (default 64)
    """

    def __init__(
        self,
        helix_hidden: int = 16,
        helix_k: int = 4,
        helix_t: int = 5,
        gnn_hidden: int = 64,
    ):
        self.helix_hidden = helix_hidden
        self.helix_k = helix_k
        self.helix_t = helix_t
        self.gnn_hidden = gnn_hidden

        self._helix_model: HelixModel | None = None
        self._best_model = None          # winner model (may differ from HELIX)
        self._best_model_name: str | None = None
        self._validation_report: ValidationReport | None = None
        self._in_dim: int | None = None

    # ── fit ────────────────────────────────────────────────────────────────────

    def fit(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        labels: np.ndarray,
        model: str = "auto",
        cfg: TrainConfig | None = None,
        edge_weight: np.ndarray | None = None,
    ) -> "HelixFramework":
        """
        Train the framework on a labeled graph dataset.

        Parameters
        ----------
        x          : (N, D) node features
        edge_index : (2, E) int array
        labels     : (N,) binary array, 1=anomalous, -1=unlabeled
        model      : 'auto' (Graph Validator picks) or one of HELIX/SAGE/GCN/MLP
        cfg        : TrainConfig; uses defaults if None
        edge_weight: (E,) optional

        Returns self for chaining.
        """
        N, D = x.shape
        self._in_dim = D
        if cfg is None:
            cfg = TrainConfig()

        X_t  = torch.tensor(x, dtype=torch.float32)
        ei_t = torch.tensor(edge_index, dtype=torch.long)
        y_t  = torch.tensor(labels, dtype=torch.float32)
        ew_t = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight is not None else None

        tr_idx, va_idx = stratified_split(labels, seed=cfg.seed)

        # ── Graph Validator ──────────────────────────────────────────────────
        if model == "auto":
            logger.info("Running Graph Validator…")
            self._validation_report = validate(x, edge_index, labels, N)
            chosen = self._validation_report.recommended_model
            logger.info("Validator selected: %s (%s)",
                        chosen, self._validation_report.confidence)
        else:
            chosen = model.upper()
            if chosen not in _MODEL_CLASSES:
                raise ValueError(f"model must be one of {list(_MODEL_CLASSES)} or 'auto'")

        # ── Always train HELIX (needed for explain() and nexus()) ────────────
        helix = HelixModel(D, hidden=self.helix_hidden,
                           cheb_k=self.helix_k, t_steps=self.helix_t)
        helix_result = train(helix, X_t, ei_t, y_t, tr_idx, va_idx, cfg, ew_t)
        self._helix_model = helix
        logger.info("HELIX trained — val AUC: %.4f", helix_result.auc)

        # ── Train winning model if different from HELIX ──────────────────────
        if chosen == "HELIX":
            self._best_model = helix
            self._best_model_name = "HELIX"
        else:
            ModelCls = _MODEL_CLASSES[chosen]
            winner = ModelCls(D, hidden=self.gnn_hidden)
            winner_result = train(winner, X_t, ei_t, y_t, tr_idx, va_idx, cfg, ew_t)
            self._best_model = winner
            self._best_model_name = chosen
            logger.info("%s trained — val AUC: %.4f", chosen, winner_result.auc)

        return self

    # ── predict ────────────────────────────────────────────────────────────────

    def predict(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fast inference using Laser Query (HELIX) or best model forward pass.
        Returns anomaly scores in [0, 1] via sigmoid.
        """
        self._check_fitted()
        X_t  = torch.tensor(x, dtype=torch.float32)
        ei_t = torch.tensor(edge_index, dtype=torch.long)
        ew_t = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight is not None else None

        self._best_model.eval()
        with torch.no_grad():
            if self._best_model_name == "HELIX":
                logits = self._helix_model.laser_query(X_t, ei_t, ew_t)
            else:
                out = self._best_model(X_t, ei_t, ew_t)
                logits = out[0] if isinstance(out, tuple) else out
            return torch.sigmoid(logits.squeeze(-1)).numpy()

    # ── explain ────────────────────────────────────────────────────────────────

    def explain(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> ExplainResult:
        """
        Full HELIX forward pass with geometric metrics.

        If the selected model is not HELIX, scores come from the best model
        but geometric metrics (rho, eta, geo_dist) always come from HELIX.

        Emits a warning when E/N > 10 or σ_seeds >= 0.025, as geometric
        metrics are less reliable in those regimes.
        """
        self._check_fitted()
        self._warn_if_incompatible()

        X_t  = torch.tensor(x, dtype=torch.float32)
        ei_t = torch.tensor(edge_index, dtype=torch.long)
        ew_t = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight is not None else None

        self._helix_model.eval()
        with torch.no_grad():
            logits_h, q_final = self._helix_model(X_t, ei_t, ew_t)

        # Use best model for scores, HELIX for geometry
        if self._best_model_name != "HELIX":
            self._best_model.eval()
            with torch.no_grad():
                out = self._best_model(X_t, ei_t, ew_t)
                logits_best = out[0] if isinstance(out, tuple) else out
            scores = torch.sigmoid(logits_best.squeeze(-1)).numpy()
        else:
            scores = torch.sigmoid(logits_h.squeeze(-1)).numpy()

        conf = self._validation_report.confidence if self._validation_report else None

        return ExplainResult(
            scores=scores,
            rho=rho(q_final).numpy(),
            eta=eta(q_final).numpy(),
            geo_dist=geo_distance(q_final).numpy(),
            model_used=self._best_model_name,
            confidence=conf,
            q_final=q_final.detach().numpy(),
        )

    # ── nexus ──────────────────────────────────────────────────────────────────

    def nexus(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        confirmed: Sequence[int],
        alpha: float = 2.0,
        edge_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        NEXUS gravitational scorer — semi-supervised propagation from confirmed anomalies.

        Does not retrain. Uses HELIX quaternion embeddings to compute geodesic
        proximity to confirmed fraud nodes in S³.

        Parameters
        ----------
        confirmed : list of node indices confirmed as anomalous
        alpha     : geodesic decay rate (higher = sharper falloff)

        Returns (N,) scores in [0, 1].
        """
        self._check_fitted()
        X_t  = torch.tensor(x, dtype=torch.float32)
        ei_t = torch.tensor(edge_index, dtype=torch.long)
        ew_t = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight is not None else None

        self._helix_model.eval()
        with torch.no_grad():
            _, q_final = self._helix_model(X_t, ei_t, ew_t)

        return nexus_score_normalized(q_final, confirmed, alpha=alpha)

    # ── validate ───────────────────────────────────────────────────────────────

    def validate(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        labels: np.ndarray,
    ) -> ValidationReport:
        """
        Run Graph Validator and return a diagnostic report.
        Does not modify the framework state — safe to call at any time.
        """
        N = x.shape[0]
        report = validate(x, edge_index, labels, N)
        return report

    # ── internals ──────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self._helix_model is None:
            raise RuntimeError("Call fit() before predict/explain/nexus.")

    def _warn_if_incompatible(self):
        if self._validation_report is None:
            return
        r = self._validation_report
        if r.graph_density > 10:
            warnings.warn(
                f"E/N={r.graph_density:.1f} > 10 (dense graph). "
                "HELIX geometric metrics (rho, tau, geo_dist) may be unreliable. "
                "Scores from the selected model are still valid.",
                UserWarning, stacklevel=3,
            )
        elif r.sigma_seeds >= 0.025:
            warnings.warn(
                f"σ_seeds={r.sigma_seeds:.4f} >= 0.025. "
                "HELIX is unstable on this graph — geometric metrics may not reflect true anomaly structure.",
                UserWarning, stacklevel=3,
            )
