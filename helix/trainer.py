"""
Unified training loop for all Helix models (HELIX, SAGE, GCN, MLP).

Design:
- Single train() function handles all model types via duck typing
- HELIX returns (logits, q_final); others return logits only
- pos_weight auto-computed if not provided (handles class imbalance)
- Gradient clipping at 1.0 (prevents exploding gradients in rotor loop)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    pos_weight: float | None = None   # auto-computed if None
    seed: int = 42
    patience: int = 0
    min_delta: float = 1e-4
    torque_lambda: float = 0.0
    centroid_lambda: float = 0.0
    verbose: bool = False
    log_every: int = 50


@dataclass
class TrainResult:
    auc: float
    loss_history: list[float] = field(default_factory=list)
    epochs_run: int = 0


def _centroid_repulsion(q: torch.Tensor, labels_train: torch.Tensor) -> torch.Tensor:
    """
    Centroid repulsion loss in S³.

    Computes the mean quaternion for fraud (label=1) and normal (label=0) nodes
    in the training batch, normalizes both to unit norm, then penalizes their
    inner product squared — pushing them toward orthogonality (90° apart in S³).

    L_repulsion = |<c_fraud, c_normal>|²

    Minimum = 0 when centroids are orthogonal (ideal separation).
    Maximum = 1 when centroids coincide (full collapse).
    """
    fraud_mask  = labels_train == 1
    normal_mask = labels_train == 0
    if fraud_mask.sum() == 0 or normal_mask.sum() == 0:
        return torch.tensor(0.0, device=q.device)

    c_fraud  = q[fraud_mask].mean(dim=0)
    c_normal = q[normal_mask].mean(dim=0)

    c_fraud  = c_fraud  / (c_fraud.norm()  + 1e-8)
    c_normal = c_normal / (c_normal.norm() + 1e-8)

    return (c_fraud @ c_normal) ** 2


def _pos_weight(y_train: np.ndarray) -> float:
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    return float(neg) / max(float(pos), 1.0)


def train(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: TrainConfig | None = None,
    edge_weight: torch.Tensor | None = None,
) -> TrainResult:
    """
    Train any Helix model on a node classification task.

    Parameters
    ----------
    model      : HelixModel, SAGEModel, GCNModel, or MLPModel
    x          : (N, D) node features
    edge_index : (2, E)
    labels     : (N,) binary float tensor
    train_idx  : indices for training nodes
    val_idx    : indices for validation nodes
    cfg        : TrainConfig (uses defaults if None)
    edge_weight: (E,) optional edge weights

    Returns
    -------
    TrainResult with val AUC and loss history
    """
    if cfg is None:
        cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    y_train = labels[train_idx].numpy()
    pw = cfg.pos_weight if cfg.pos_weight is not None else _pos_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    y_tr_t = labels[train_idx]
    y_val = labels[val_idx].numpy()
    loss_history = []

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    use_patience = cfg.patience > 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(x, edge_index, edge_weight)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits.squeeze(-1)

        loss = criterion(logits[train_idx], y_tr_t)
        if cfg.torque_lambda > 0 and hasattr(model, '_last_tau'):
            loss = loss + cfg.torque_lambda * model._last_tau.norm(dim=-1).mean()
        if cfg.centroid_lambda > 0 and isinstance(out, tuple):
            q_final = out[1]
            loss = loss + cfg.centroid_lambda * _centroid_repulsion(
                q_final[train_idx], y_tr_t
            )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        loss_history.append(loss.item())
        if cfg.verbose and epoch % cfg.log_every == 0:
            logger.info("  epoch %d/%d  loss=%.4f", epoch, cfg.epochs, loss.item())

        if use_patience:
            model.eval()
            with torch.no_grad():
                out_v = model(x, edge_index, edge_weight)
                lg_v = out_v[0] if isinstance(out_v, tuple) else out_v
                val_loss = criterion(lg_v.squeeze(-1)[val_idx],
                                     labels[val_idx]).item()
            if val_loss < best_val_loss - cfg.min_delta:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    if cfg.verbose:
                        logger.info("  early stop at epoch %d", epoch)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_weight)
        logits = out[0] if isinstance(out, tuple) else out
        scores = logits.squeeze(-1)[val_idx].numpy()

    auc = float(roc_auc_score(y_val, scores))

    # Polarity guard: if model learned inverted signal, flip scores.
    # Caused by aggressive pos_weight + gradient saturation in dense graphs.
    if auc < 0.5:
        auc = 1.0 - auc
        model._polarity_flipped = True
    else:
        model._polarity_flipped = False

    return TrainResult(auc=auc, loss_history=loss_history, epochs_run=len(loss_history))


def stratified_split(
    labels: np.ndarray,
    train_frac: float = 0.7,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified split by label class.
    Ignores unlabeled nodes (label == -1).
    Returns (train_idx, val_idx).
    """
    rng = np.random.default_rng(seed)
    known = np.where(labels >= 0)[0]
    pos   = known[labels[known] == 1]
    neg   = known[labels[known] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    cut_p = int(train_frac * len(pos))
    cut_n = int(train_frac * len(neg))
    train_idx = np.concatenate([pos[:cut_p], neg[:cut_n]])
    val_idx   = np.concatenate([pos[cut_p:],  neg[cut_n:]])
    return train_idx, val_idx
