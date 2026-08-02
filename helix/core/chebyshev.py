"""
Chebyshev spectral field propagator.
Complexity: O(K * |E|) — no eigendecomposition.
"""

import torch
import torch.nn as nn


def chebyshev_propagate(
    L_tilde: torch.Tensor,
    x: torch.Tensor,
    thetas: torch.Tensor,
) -> torch.Tensor:
    """
    Chebyshev expansion: Φ = Σ_{k=0}^{K} θ_k T_k(L_tilde) x
    x: (N, C), thetas: (K+1,). Returns (N, C).
    """
    K = thetas.shape[0] - 1
    T_prev = x
    T_curr = torch.sparse.mm(L_tilde, x)
    out = thetas[0] * T_prev + thetas[1] * T_curr
    for k in range(2, K + 1):
        T_next = 2 * torch.sparse.mm(L_tilde, T_curr) - T_prev
        out = out + thetas[k] * T_next
        T_prev, T_curr = T_curr, T_next
    return out


class ChebyshevFNO(nn.Module):
    """
    Learnable Chebyshev field operator over a 3D vector field.
    One independent theta vector per channel.

    residual_mix: if True, output is α·Φ_propagated + (1−α)·x_local where α
    is a learned sigmoid parameter. Prevents over-smoothing in dense graphs —
    the model learns to down-weight neighbour signals when they add noise.
    """

    def __init__(self, order: int = 4, num_channels: int = 3, residual_mix: bool = True):
        super().__init__()
        self.order = order
        self.num_channels = num_channels
        self.residual_mix = residual_mix
        self.thetas = nn.Parameter(torch.randn(order + 1, num_channels) * 0.1)
        if residual_mix:
            # initialise near 0.9 so sparse-graph behaviour is preserved by default
            self.alpha_logit = nn.Parameter(torch.tensor(2.2))

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """x: (N, 3), L_tilde: sparse scaled Laplacian. Returns (N, 3)."""
        propagated = torch.cat([
            chebyshev_propagate(L_tilde, x[:, c:c+1], self.thetas[:, c])
            for c in range(self.num_channels)
        ], dim=-1)
        if self.residual_mix:
            alpha = torch.sigmoid(self.alpha_logit)
            return alpha * propagated + (1.0 - alpha) * x
        return propagated
