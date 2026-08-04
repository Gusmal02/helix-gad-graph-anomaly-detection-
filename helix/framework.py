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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from helix.models.helix import HelixModel
from helix.models.sage  import SAGEModel
from helix.models.gcn   import GCNModel
from helix.models.mlp   import MLPModel
from helix.trainer import TrainConfig, TrainResult, train, stratified_split
from helix.validator import validate, ValidationReport
from helix.nexus import nexus_score_normalized, sonar_score
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
    torque: np.ndarray          # (N,) ‖τ‖ — torque magnitude (geometric gradient)
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
        directed: bool = False,
        spectral_norm: bool = True,
    ):
        self.helix_hidden = helix_hidden
        self.helix_k = helix_k
        self.helix_t = helix_t
        self.gnn_hidden = gnn_hidden
        self.directed = directed
        self.spectral_norm = spectral_norm

        self._helix_model: HelixModel | None = None
        self._best_model = None
        self._best_model_name: str | None = None
        self._validation_report: ValidationReport | None = None
        self._in_dim: int | None = None
        self._pca: bool = False
        self._pca_components: np.ndarray | None = None
        self._pca_mean: np.ndarray | None = None

    # ── fit ────────────────────────────────────────────────────────────────────

    def fit(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        labels: np.ndarray,
        model: str = "auto",
        cfg: TrainConfig | None = None,
        edge_weight: np.ndarray | None = None,
        auto_pca: int | None = None,
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
        auto_pca   : if set, reduce features to this many PCA components before training

        Returns self for chaining.
        """
        N, D = x.shape
        self._in_dim = D

        if auto_pca is not None and auto_pca < D:
            self._pca_mean = x.mean(axis=0)
            centered = x - self._pca_mean
            _, s, Vt = np.linalg.svd(centered, full_matrices=False)
            self._pca_components = Vt[:auto_pca]
            self._pca = True
            x = centered @ self._pca_components.T
            D = auto_pca
            logger.info("Auto-PCA: %d → %d components", self._in_dim, D)
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
                           cheb_k=self.helix_k, t_steps=self.helix_t,
                           directed=self.directed, spectral_norm=self.spectral_norm)
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
        x = self._pca_transform(x)
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
            scores = torch.sigmoid(logits.squeeze(-1)).numpy()
        if getattr(self._best_model, '_polarity_flipped', False):
            scores = 1.0 - scores
        return scores

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
        x = self._pca_transform(x)

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
            if getattr(self._best_model, '_polarity_flipped', False):
                scores = 1.0 - scores
        else:
            scores = torch.sigmoid(logits_h.squeeze(-1)).numpy()
            if getattr(self._helix_model, '_polarity_flipped', False):
                scores = 1.0 - scores

        conf = self._validation_report.confidence if self._validation_report else None
        tau = self._helix_model._last_tau
        torque_mag = tau.norm(dim=-1).numpy()

        return ExplainResult(
            scores=scores,
            rho=rho(q_final).numpy(),
            eta=eta(q_final).numpy(),
            geo_dist=geo_distance(q_final).numpy(),
            torque=torque_mag,
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
        alpha: float | str = 2.0,
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
        x = self._pca_transform(x)
        X_t  = torch.tensor(x, dtype=torch.float32)
        ei_t = torch.tensor(edge_index, dtype=torch.long)
        ew_t = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight is not None else None

        self._helix_model.eval()
        with torch.no_grad():
            _, q_final = self._helix_model(X_t, ei_t, ew_t)

        return nexus_score_normalized(q_final, confirmed, alpha=alpha)

    # ── sonar ──────────────────────────────────────────────────────────────────

    def sonar(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        confirmed: Sequence[int],
        alpha: float | str = 2.0,
        max_hops: int = 4,
        hop_decay: float = 0.6,
        edge_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        SONAR scorer — combines S³ proximity (NEXUS) with BFS hop distance.

        Nodes that are both geometrically close in S³ AND few hops from
        confirmed seeds score highest.

        Parameters
        ----------
        confirmed : list of node indices confirmed as anomalous
        alpha     : geodesic decay rate, or 'auto' to calibrate from data
        max_hops  : BFS depth limit
        hop_decay : multiplicative penalty per hop (0 < hop_decay <= 1)

        Returns (N,) scores in [0, 1].
        """
        self._check_fitted()
        x = self._pca_transform(x)
        X_t  = torch.tensor(x, dtype=torch.float32)
        ei_t = torch.tensor(edge_index, dtype=torch.long)
        ew_t = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight is not None else None

        self._helix_model.eval()
        with torch.no_grad():
            _, q_final = self._helix_model(X_t, ei_t, ew_t)

        return sonar_score(q_final, ei_t, confirmed, alpha=alpha,
                           max_hops=max_hops, hop_decay=hop_decay)

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

    # ── save / load ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save trained framework to a file."""
        self._check_fitted()
        checkpoint = {
            "helix_state": self._helix_model.state_dict(),
            "best_model_name": self._best_model_name,
            "in_dim": self._in_dim,
            "hyper": {
                "helix_hidden": self.helix_hidden,
                "helix_k": self.helix_k,
                "helix_t": self.helix_t,
                "gnn_hidden": self.gnn_hidden,
                "directed": self.directed,
                "spectral_norm": self.spectral_norm,
            },
            "validation_report": asdict(self._validation_report) if self._validation_report else None,
            "helix_polarity": getattr(self._helix_model, '_polarity_flipped', False),
        }
        if self._best_model_name != "HELIX":
            checkpoint["best_model_state"] = self._best_model.state_dict()
            checkpoint["best_polarity"] = getattr(self._best_model, '_polarity_flipped', False)
        if self._pca:
            checkpoint["pca_components"] = self._pca_components
            checkpoint["pca_mean"] = self._pca_mean
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> "HelixFramework":
        """Load a trained framework from a file."""
        ck = torch.load(path, weights_only=False)
        h = ck["hyper"]
        fw = cls(helix_hidden=h["helix_hidden"], helix_k=h["helix_k"],
                 helix_t=h["helix_t"], gnn_hidden=h["gnn_hidden"],
                 directed=h.get("directed", False),
                 spectral_norm=h.get("spectral_norm", True))
        fw._in_dim = ck["in_dim"]

        if "pca_components" in ck:
            fw._pca_components = ck["pca_components"]
            fw._pca_mean = ck["pca_mean"]
            fw._pca = True

        D = fw._pca_components.shape[0] if fw._pca and fw._pca_components is not None else fw._in_dim
        fw._helix_model = HelixModel(D, hidden=h["helix_hidden"],
                                     cheb_k=h["helix_k"], t_steps=h["helix_t"],
                                     directed=h.get("directed", False),
                                     spectral_norm=h.get("spectral_norm", True))
        fw._helix_model.load_state_dict(ck["helix_state"])
        fw._helix_model._polarity_flipped = ck.get("helix_polarity", False)

        fw._best_model_name = ck["best_model_name"]
        if fw._best_model_name == "HELIX":
            fw._best_model = fw._helix_model
        else:
            ModelCls = _MODEL_CLASSES[fw._best_model_name]
            fw._best_model = ModelCls(D, hidden=h["gnn_hidden"])
            fw._best_model.load_state_dict(ck["best_model_state"])
            fw._best_model._polarity_flipped = ck.get("best_polarity", False)

        if ck.get("validation_report"):
            fw._validation_report = ValidationReport(**ck["validation_report"])

        return fw

    # ── internals ──────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self._helix_model is None:
            raise RuntimeError("Call fit() before predict/explain/nexus.")

    def _pca_transform(self, x: np.ndarray) -> np.ndarray:
        if self._pca:
            return (x - self._pca_mean) @ self._pca_components.T
        return x

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
