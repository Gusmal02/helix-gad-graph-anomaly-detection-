"""
Elliptic Bitcoin fraud detection demo.

Uses the pre-processed Elliptic dataset from the research project.
Expected: AUC ≈ 0.96 (Diag#42 benchmark).

Requires:
  - C:\\Users\\Gustavo\\Documents\\tripleten\\proyectos\\
    GFCN -Graph Fourier Clifford Network\\gfcn\\shared_d20.py
  - Elliptic raw data processed via that module

Run from repo root:
  python examples/elliptic_demo.py
"""

import sys
import pathlib
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# ── Locate research project for data loading ──────────────────────────────────
RESEARCH_ROOT = pathlib.Path(
    r"C:\Users\Gustavo\Documents\tripleten\proyectos"
    r"\GFCN -Graph Fourier Clifford Network"
)
if not RESEARCH_ROOT.exists():
    raise SystemExit(f"Research project not found at:\n  {RESEARCH_ROOT}")

sys.path.insert(0, str(RESEARCH_ROOT))

from gfcn.shared_d20 import load_elliptic  # type: ignore

from helix import HelixFramework
from helix.trainer import TrainConfig
from helix.metrics import best_f1_geo
import torch

# ── Load Elliptic ─────────────────────────────────────────────────────────────
print("Loading Elliptic (D=20)…")
labels, timestep, X_np, edge_np, _ = load_elliptic(d=20)
labels = labels.astype(np.int32)
N = len(labels)

print(f"  N={N} nodes  |  known labeled: {(labels >= 0).sum()}")
print(f"  Illicit: {(labels == 1).sum()}  Licit: {(labels == 0).sum()}")
print(f"  E={edge_np.shape[1]}  E/N={edge_np.shape[1]/N:.2f}")

# ── Train HELIX ───────────────────────────────────────────────────────────────
fw = HelixFramework(helix_hidden=16, helix_k=4, helix_t=5)

cfg = TrainConfig(epochs=200, lr=1e-3, seed=42, verbose=True, log_every=50)

print("\nTraining HELIX on Elliptic…")
fw.fit(X_np, edge_np, labels.astype(np.float32), model="HELIX", cfg=cfg)

# ── Evaluate ──────────────────────────────────────────────────────────────────
known_mask = labels >= 0
X_known    = X_np[known_mask]
ei_known   = edge_np   # use full graph (transductive)
y_known    = labels[known_mask]

scores = fw.predict(X_np, edge_np)[known_mask]
auc    = roc_auc_score(y_known, scores)
print(f"\nAUC-ROC : {auc:.4f}  (benchmark ≈ 0.9624)")

threshold = 0.5
preds = (scores >= threshold).astype(int)
print(f"\nClassification report (threshold={threshold}):")
print(classification_report(y_known, preds, target_names=["licit", "illicit"]))

# ── Geometric metrics ─────────────────────────────────────────────────────────
result = fw.explain(X_np, edge_np)
q_final = torch.tensor(result.q_final)
y_tensor = torch.tensor(labels.astype(np.float32))

f1_geo, best_theta = best_f1_geo(
    q_final[known_mask],
    y_tensor[known_mask].long(),
)
print(f"F1_geo  : {f1_geo:.4f}  (best θ={best_theta:.2f} rad)")

fraud_idx = np.where(labels == 1)[0]
legit_idx  = np.where(labels == 0)[0]
print(f"\nGeometric separation:")
print(f"  ρ mean — illicit: {result.rho[fraud_idx].mean():.4f}  "
      f"licit: {result.rho[legit_idx].mean():.4f}")
print(f"  geo_dist — illicit: {result.geo_dist[fraud_idx].mean():.4f}  "
      f"licit: {result.geo_dist[legit_idx].mean():.4f}")

# ── NEXUS from 10 confirmed illicit nodes ────────────────────────────────────
confirmed = fraud_idx[:10].tolist()
nexus = fw.nexus(X_np, edge_np, confirmed=confirmed)
remaining = set(fraud_idx[10:].tolist())
for k in [50, 100, 200]:
    topk = set(np.argsort(nexus)[-k:].tolist())
    recall = len(remaining & topk) / max(len(remaining), 1)
    print(f"  NEXUS recall@{k:<3}: {recall:.2%}")
