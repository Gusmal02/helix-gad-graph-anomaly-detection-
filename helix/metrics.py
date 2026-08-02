"""
Geometric metrics native to HELIX — computed from quaternion state q_final.

rho   (ρ): imaginary norm — local instability / anomaly signal
eta   (η): scalar part — alignment with identity rotor
tau_norm: cross-product magnitude — field torque intensity
geo_flag: True if node did not rotate in S³ (anomalous geometry)
sigma_seeds: inter-seed AUC std — used by Graph Validator
"""

import torch
import numpy as np
from typing import Sequence


def rho(q_final: torch.Tensor) -> torch.Tensor:
    """ρ = ||q_imag|| per node. Shape: (N,). Higher → more anomalous."""
    return q_final[:, 1:].norm(dim=-1)


def eta(q_final: torch.Tensor) -> torch.Tensor:
    """η = q_real (scalar part). Shape: (N,). Close to 1 → near identity."""
    return q_final[:, 0]


def geo_distance(q_final: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance from identity quaternion on S³.
    dist = arccos(|w|). Shape: (N,).
    """
    w = q_final[:, 0].clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(w.abs())


def geo_flag(q_final: torch.Tensor, theta: float) -> torch.Tensor:
    """
    Boolean mask: True where geodesic distance > theta.
    Nodes that did NOT rotate back to identity → geometrically anomalous.
    Shape: (N,).
    """
    return geo_distance(q_final) > theta


def best_f1_geo(
    q_final: torch.Tensor,
    labels: torch.Tensor,
    thetas: Sequence[float] | None = None,
) -> tuple[float, float]:
    """
    Sweep theta values and return (best_F1_geo, best_theta).
    labels: binary (N,) tensor, 1 = anomalous.
    """
    if thetas is None:
        thetas = torch.linspace(0.05, 1.5, 20).tolist()

    q_np = q_final.detach().cpu()
    y = labels.cpu().numpy()
    best_f1, best_theta = 0.0, 0.0

    for th in thetas:
        pred = geo_flag(q_np, th).numpy().astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_theta = float(f1), float(th)

    return best_f1, best_theta


def tau_ratio(
    q_final: torch.Tensor,
    phi_base: torch.Tensor,
    phi_prop: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    ratio_τ = mean_tau(ill) / mean_tau(lit).
    Values > 1 mean ill nodes generate more torque than lit.
    Natural baseline (no regularization) ≈ 0.4–0.7.
    """
    tau = torch.cross(phi_base, phi_prop, dim=-1).norm(dim=-1).detach().cpu()
    y   = labels.cpu()
    ill_mask = y == 1
    lit_mask = y == 0
    mean_ill = tau[ill_mask].mean().item() if ill_mask.any() else 0.0
    mean_lit = tau[lit_mask].mean().item() if lit_mask.any() else 1e-9
    return mean_ill / (mean_lit + 1e-9)


def sigma_seeds(auc_list: list[float]) -> float:
    """
    Inter-seed AUC standard deviation.
    < 0.015: HELIX (high confidence)
    0.015–0.025: HELIX (medium confidence)
    >= 0.025: unstable → SAGE or MLP
    """
    return float(np.std(auc_list))
