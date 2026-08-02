"""
Quick start — Helix on a synthetic fraud graph.

Demonstrates the full API in ~40 lines:
  fit → predict → explain → nexus → validate

No real datasets required. Run from the repo root:
  python examples/quick_start.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from helix import HelixFramework
from helix.trainer import TrainConfig

# ── 1. Synthetic fraud graph ──────────────────────────────────────────────────
# N=500 nodes (accounts), E=800 causal edges, D=12 features
# Fraud rate 8% — typical for financial fraud domains

rng = np.random.default_rng(42)
N, D, E = 500, 12, 800

# Features: fraudulent nodes have slightly different distributions
labels = rng.choice([0, 1], N, p=[0.92, 0.08]).astype(np.float32)
x = rng.standard_normal((N, D)).astype(np.float32)
x[labels == 1] += rng.standard_normal((int(labels.sum()), D)) * 0.5

# Sparse causal graph (E/N=1.6 → HELIX regime)
src = rng.integers(0, N, E)
dst = rng.integers(0, N, E)
edge_index = np.stack([src, dst]).astype(np.int64)

print(f"Graph: N={N} nodes, E={E} edges, E/N={E/N:.1f}, fraud rate={labels.mean():.1%}")

# ── 2. Framework ──────────────────────────────────────────────────────────────
fw = HelixFramework(helix_hidden=16, helix_k=4, helix_t=5)

cfg = TrainConfig(epochs=100, lr=1e-3, seed=42, verbose=True, log_every=25)
fw.fit(x, edge_index, labels, model="HELIX", cfg=cfg)

# ── 3. predict() — fast inference (Laser Query) ───────────────────────────────
scores = fw.predict(x, edge_index)
auc = roc_auc_score(labels, scores)
print(f"\npredict() AUC: {auc:.4f}")

# ── 4. explain() — full loop + geometric metrics ──────────────────────────────
result = fw.explain(x, edge_index)

fraud_idx = np.where(labels == 1)[0]
legit_idx  = np.where(labels == 0)[0]

print(f"\nexplain() model: {result.model_used}")
print(f"  ρ (instability)  — fraud: {result.rho[fraud_idx].mean():.4f}  "
      f"legit: {result.rho[legit_idx].mean():.4f}")
print(f"  η (identity aln) — fraud: {result.eta[fraud_idx].mean():.4f}  "
      f"legit: {result.eta[legit_idx].mean():.4f}")
print(f"  geo_dist mean    — fraud: {result.geo_dist[fraud_idx].mean():.4f}  "
      f"legit: {result.geo_dist[legit_idx].mean():.4f}")

# ── 5. nexus() — semi-supervised from confirmed fraud ────────────────────────
# Suppose operations confirmed 5 fraud cases
confirmed = fraud_idx[:5].tolist()
nexus_scores = fw.nexus(x, edge_index, confirmed=confirmed, alpha=2.0)

# How many of the remaining fraud nodes rank in top-50?
remaining_fraud = set(fraud_idx[5:].tolist())
top50 = set(np.argsort(nexus_scores)[-50:].tolist())
recall_at_50 = len(remaining_fraud & top50) / max(len(remaining_fraud), 1)
print(f"\nnexus() recall@50 (non-confirmed fraud): {recall_at_50:.2%}")

# ── 6. validate() — graph diagnostics ────────────────────────────────────────
print("\nRunning Graph Validator (this takes ~30s)…")
report = fw.validate(x, edge_index, labels.astype(np.int32))
print(f"  Recommended model : {report.recommended_model} ({report.confidence})")
print(f"  Reason            : {report.reason}")
print(f"  σ_seeds           : {report.sigma_seeds:.4f}")
print(f"  E/N density       : {report.graph_density:.2f}")
print(f"  Gini coefficient  : {report.gini:.3f}")
print(f"  HELIX AUC mean    : {report.helix_auc_mean:.4f}")
print(f"  MLP AUC           : {report.mlp_auc:.4f}")
print(f"  Lift (HELIX-MLP)  : {report.lift:+.4f}")
