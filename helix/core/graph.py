"""
Graph structural diagnostics used by the Graph Validator.
All functions operate on (edge_index, num_nodes) — no external graph libraries.
"""

import torch
import numpy as np


def edge_density(edge_index: torch.Tensor, num_nodes: int) -> float:
    """E/N ratio — primary predictor of HELIX vs SAGE regime."""
    return edge_index.shape[1] / max(num_nodes, 1)


def degree_gini(edge_index: torch.Tensor, num_nodes: int) -> float:
    """
    Gini coefficient of the degree distribution.
    High Gini (> 0.6) indicates hub-and-spoke structure → SAGE wins.
    """
    row = edge_index[0]
    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones(row.shape[0]))
    deg_np = deg.numpy()
    if deg_np.sum() == 0:
        return 0.0
    sorted_d = np.sort(deg_np)
    n = len(sorted_d)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * sorted_d).sum() - (n + 1) * sorted_d.sum()) /
                 (n * sorted_d.sum() + 1e-9))


def label_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fraction of edges connecting nodes with the same label.
    High homophily (> 0.95) with high Gini → SAGE wins (community detection advantage).
    """
    row, col = edge_index
    if row.shape[0] == 0:
        return 0.0
    same = (labels[row] == labels[col]).float().mean().item()
    return float(same)


def sparsify_top_k(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    k: int = 10,
    num_nodes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keep at most k neighbours per node (highest edge weight, or arbitrary if uniform).
    Reduces effective E/N before FNO propagation, preventing over-smoothing in dense graphs.

    Returns (sparse_edge_index, sparse_edge_weight).
    """
    N = num_nodes or int(edge_index.max().item()) + 1
    src, dst = edge_index
    ew = edge_weight if edge_weight is not None else torch.ones(src.shape[0])

    keep = torch.zeros(src.shape[0], dtype=torch.bool)
    for node in range(N):
        mask = src == node
        idx  = mask.nonzero(as_tuple=True)[0]
        if idx.numel() <= k:
            keep[idx] = True
        else:
            weights = ew[idx]
            top_k   = weights.topk(k).indices
            keep[idx[top_k]] = True

    return edge_index[:, keep], ew[keep]


def graph_stats(
    edge_index: torch.Tensor,
    num_nodes: int,
    labels: torch.Tensor | None = None,
) -> dict:
    """Returns a summary dict consumed by the Graph Validator."""
    stats = {
        "N": num_nodes,
        "E": edge_index.shape[1],
        "density": edge_density(edge_index, num_nodes),
        "gini": degree_gini(edge_index, num_nodes),
    }
    if labels is not None:
        stats["homophily"] = label_homophily(edge_index, labels)
    else:
        stats["homophily"] = None
    return stats
