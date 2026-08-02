"""
Normalized graph Laplacian for Chebyshev propagation.
Supports directed graphs by using symmetric degree (in + out).
"""

import torch


def normalized_laplacian(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, float]:
    """
    Returns (L_tilde, lambda_max).
    L_sym   = I - D^{-1/2} A D^{-1/2}     eigenvalues ∈ [0, 2]
    L_tilde = 2/lambda_max * L_sym - I     eigenvalues ∈ [-1, 1]

    Directed graphs: degree = in-degree + out-degree (symmetric treatment)
    to avoid NaN from isolated sinks/sources.
    """
    row, col = edge_index
    device = edge_weight.device

    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row, edge_weight)   # out-degree
    deg.scatter_add_(0, col, edge_weight)   # in-degree  (symmetric treatment)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0.0

    norm_w = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    diag_idx = torch.arange(num_nodes, device=device)
    diag_inds = torch.stack([diag_idx, diag_idx])
    edge_inds = torch.stack([row, col])

    # L_sym = I - A_norm
    L_sym = torch.sparse_coo_tensor(
        torch.cat([diag_inds, edge_inds], dim=1),
        torch.cat([torch.ones(num_nodes, device=device), -norm_w]),
        (num_nodes, num_nodes),
    ).coalesce()

    # Estimate lambda_max via power iteration
    v = torch.randn(num_nodes, 1, device=device)
    for _ in range(10):
        v = torch.sparse.mm(L_sym, v)
        nv = v.norm()
        if nv < 1e-10:
            break
        v = v / nv
    lambda_max = float((v.t() @ torch.sparse.mm(L_sym, v)).item())
    lambda_max = max(lambda_max * 1.05, 1.0)

    scale = 2.0 / lambda_max
    L_tilde = torch.sparse_coo_tensor(
        torch.cat([diag_inds, edge_inds], dim=1),
        torch.cat([
            torch.full((num_nodes,), scale - 1.0, device=device),
            -scale * norm_w,
        ]),
        (num_nodes, num_nodes),
    ).coalesce()

    return L_tilde, lambda_max
