"""
Quaternion rotor arithmetic — operates in S³.
Convention: q = (w, x, y, z) where w is the scalar part, ||q|| = 1.
"""

import torch
import torch.nn as nn


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)


def quat_exp(v: torch.Tensor) -> torch.Tensor:
    """
    Quaternion exponential of a pure imaginary vector v ∈ R³.
    exp(v) = cos(||v||) + sin(||v||) * v/||v||
    Input: (..., 3). Output: (..., 4) as (w, x, y, z).
    """
    angle = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = v / angle
    return torch.cat([torch.cos(angle), torch.sin(angle) * axis], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product. Shape: (..., 4)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, xyz = q[..., :1], q[..., 1:]
    return torch.cat([w, -xyz], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate v ∈ R³ by unit quaternion q: v' = q v q*.
    q: (..., 4), v: (..., 3). Returns (..., 3).
    """
    v_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    rotated = quat_mul(quat_mul(q, v_quat), quat_conjugate(q))
    return rotated[..., 1:]


class RotorStep(nn.Module):
    """
    Single rotor step — shared parameters across all nodes (inductive).
    Given q (N,4), field tau (N,3), coupling eta (N,):
      1. Free drift:  q_free = exp(omega * dt/2 * axis) ⊗ q
      2. Coupling:    q_new  = exp(eta * ||tau||/2 * tau/||tau||) ⊗ q_free
    No .detach() inside — caller controls gradient flow.
    """

    def __init__(self, dt: float = 1.0):
        super().__init__()
        self.dt = dt
        self.omega = nn.Parameter(torch.tensor(0.1))
        self.axis_raw = nn.Parameter(torch.randn(3))

    @property
    def axis(self) -> torch.Tensor:
        return self.axis_raw / (self.axis_raw.norm() + 1e-8)

    def forward(self, q: torch.Tensor,
                tau: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        N = q.shape[0]
        angle_axis = (self.omega * self.dt / 2) * self.axis
        delta_q = quat_exp(angle_axis.unsqueeze(0).expand(N, -1))
        q_free = quat_normalize(quat_mul(delta_q, q))

        tau_norm = tau.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        n_hat = tau / tau_norm
        angle = eta.unsqueeze(-1) * tau_norm / 2
        q_align = quat_exp(angle * n_hat)
        return quat_normalize(quat_mul(q_align, q_free))
