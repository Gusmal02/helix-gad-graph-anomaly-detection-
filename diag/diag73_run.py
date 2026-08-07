"""
Diag#73 — Ablación de regularizadores en Elliptic (Grupos A, B, C).
Grupo D se configura manualmente tras ver resultados de B y C.
"""
import sys, os
sys.path.insert(0, r"C:\Users\Gustavo\Documents\tripleten\proyectos\helix")
os.chdir(r"C:\Users\Gustavo\Documents\tripleten\proyectos\helix")

import time
import json
import pickle
import numpy as np
import torch

from helix import HelixFramework
from helix.trainer import TrainConfig

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\Gustavo\Documents\tripleten\proyectos\GFCN -Graph Fourier Clifford Network\data\processed\funnel_elliptic.pkl"
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

X          = data["X3"]
ei         = data["edge_index"]
labels_raw = data["labels"]
labels     = labels_raw.astype(np.float32)

BASELINE_AUC = 0.9624

RUNS = [
    # Grupo A — ablación spectral_norm
    {"run": "A1", "spectral_norm": True,  "torque_lambda": 0.00, "centroid_lambda": 0.00, "directed": False, "auto_pca": None},
    {"run": "A2", "spectral_norm": False, "torque_lambda": 0.00, "centroid_lambda": 0.00, "directed": False, "auto_pca": None},
    # Grupo B — sweep torque_lambda
    {"run": "B1", "spectral_norm": True,  "torque_lambda": 0.01, "centroid_lambda": 0.00, "directed": False, "auto_pca": None},
    {"run": "B2", "spectral_norm": True,  "torque_lambda": 0.05, "centroid_lambda": 0.00, "directed": False, "auto_pca": None},
    {"run": "B3", "spectral_norm": True,  "torque_lambda": 0.10, "centroid_lambda": 0.00, "directed": False, "auto_pca": None},
    {"run": "B4", "spectral_norm": True,  "torque_lambda": 0.20, "centroid_lambda": 0.00, "directed": False, "auto_pca": None},
    # Grupo C — sweep centroid_lambda
    {"run": "C1", "spectral_norm": True,  "torque_lambda": 0.00, "centroid_lambda": 0.05, "directed": False, "auto_pca": None},
    {"run": "C2", "spectral_norm": True,  "torque_lambda": 0.00, "centroid_lambda": 0.10, "directed": False, "auto_pca": None},
    {"run": "C3", "spectral_norm": True,  "torque_lambda": 0.00, "centroid_lambda": 0.20, "directed": False, "auto_pca": None},
    {"run": "C4", "spectral_norm": True,  "torque_lambda": 0.00, "centroid_lambda": 0.50, "directed": False, "auto_pca": None},
]

results = []

for cfg_d in RUNS:
    print(f"\n{'='*55}")
    print(f"Run {cfg_d['run']} | sn={cfg_d['spectral_norm']} "
          f"torque={cfg_d['torque_lambda']} centroid={cfg_d['centroid_lambda']}")

    cfg = TrainConfig(
        epochs=200,
        lr=1e-3,
        grad_clip=1.0,
        seed=42,
        torque_lambda=cfg_d["torque_lambda"],
        centroid_lambda=cfg_d["centroid_lambda"],
        verbose=False,
    )
    fw = HelixFramework(
        helix_hidden=16,
        helix_k=4,
        helix_t=5,
        spectral_norm=cfg_d["spectral_norm"],
        directed=cfg_d["directed"],
    )

    t0 = time.perf_counter()
    fw.fit(X, ei, labels, model="HELIX", cfg=cfg, auto_pca=cfg_d["auto_pca"])
    elapsed = time.perf_counter() - t0

    auc = fw._train_result.auc if hasattr(fw, "_train_result") and fw._train_result else None

    result_explain = fw.explain(X, ei, labels=labels)

    fraud_mask = labels == 1
    row = {
        "run":            cfg_d["run"],
        "auc":            round(auc, 4) if auc else None,
        "delta_vs_base":  round(auc - BASELINE_AUC, 4) if auc else None,
        "epochs_run":     fw._train_result.epochs_run if hasattr(fw, "_train_result") and fw._train_result else None,
        "torque_mean":    round(float(result_explain.torque.mean()), 4),
        "geo_dist_fraud": round(float(result_explain.geo_dist[fraud_mask].mean()), 4) if fraud_mask.any() else None,
        "time_s":         round(elapsed, 1),
    }
    results.append(row)
    print(json.dumps(row, indent=2))

out_path = "diag/diag73_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*55}")
print(f"Diag#73 Grupos A-C completo. Resultados en {out_path}")
print(f"\nResumen:")
print(f"{'Run':<5} {'AUC':>7} {'Delta':>8} {'Epochs':>7} {'Time_s':>7}")
for r in results:
    auc_s   = f"{r['auc']:.4f}" if r['auc'] else "N/A"
    delta_s = f"{r['delta_vs_base']:+.4f}" if r['delta_vs_base'] else "N/A"
    print(f"{r['run']:<5} {auc_s:>7} {delta_s:>8} {str(r['epochs_run']):>7} {r['time_s']:>7}")
