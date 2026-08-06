# Diag#73 — Ablación de Regularizadores en Elliptic

**Objetivo:** Medir el impacto individual y combinado de los 3 nuevos regularizadores
(`spectral_norm`, `torque_lambda`, `centroid_lambda`) sobre el baseline AUC=0.9624
en Bitcoin Elliptic. Determinar si alguno mejora significativamente y cuál es la
combinación óptima.

**Dataset:** Elliptic Bitcoin (`funnel_elliptic.pkl`)
**Métrica principal:** AUC-ROC en val set (misma partición que Diag#63)
**Threshold de mejora significativa:** > 0.3pp sobre baseline (0.9624)

---

## Configuración base

```python
BASE_CFG = TrainConfig(
    epochs     = 200,
    lr         = 1e-3,
    grad_clip  = 1.0,
    seed       = 42,
    # todos los regularizadores en OFF
    torque_lambda   = 0.0,
    centroid_lambda = 0.0,
)
BASE_FW = HelixFramework(
    helix_hidden = 16,
    helix_k      = 4,
    helix_t      = 5,
    spectral_norm = True,   # ON por defecto — baseline real
)
```

---

## Experimentos (13 runs)

### Grupo A — Ablación spectral_norm (2 runs)

| Run | spectral_norm | torque_λ | centroid_λ | Hipótesis |
|-----|--------------|----------|------------|-----------|
| A1  | **True** (baseline) | 0.0 | 0.0 | Referencia |
| A2  | **False** | 0.0 | 0.0 | ¿Cuánto aporta la SN? |

### Grupo B — Sweep torque_lambda (4 runs)

| Run | spectral_norm | torque_λ | centroid_λ | Hipótesis |
|-----|--------------|----------|------------|-----------|
| B1  | True | **0.01** | 0.0 | λ conservador |
| B2  | True | **0.05** | 0.0 | λ moderado |
| B3  | True | **0.10** | 0.0 | λ agresivo |
| B4  | True | **0.20** | 0.0 | λ muy agresivo |

### Grupo C — Sweep centroid_lambda (4 runs)

| Run | spectral_norm | torque_λ | centroid_λ | Hipótesis |
|-----|--------------|----------|------------|-----------|
| C1  | True | 0.0 | **0.05** | λ conservador |
| C2  | True | 0.0 | **0.10** | λ moderado |
| C3  | True | 0.0 | **0.20** | λ agresivo |
| C4  | True | 0.0 | **0.50** | λ muy agresivo |

### Grupo D — Mejor combinación (3 runs)

Lanzar solo después de identificar los mejores λ en B y C.

| Run | Config | Descripción |
|-----|--------|-------------|
| D1  | best_torque + best_centroid | Combinación aditiva |
| D2  | best_torque + best_centroid + directed=True | Dirigido (Elliptic tiene dirección sender→receiver) |
| D3  | best_combo + auto_pca=20 | PCA + mejor combo |

---

## Métricas a registrar por run

```python
{
    "run":             str,       # "A1", "B2", etc.
    "auc":             float,     # AUC-ROC val
    "delta_vs_base":  float,      # auc - 0.9624
    "epochs_run":      int,       # early stop o 200
    "loss_final":      float,     # última loss del loop
    "torque_mean":     float,     # mean(‖τ‖) al final — desde result.torque via explain()
    "centroid_inner":  float,     # |⟨c_fraud, c_normal⟩| al final — indicador de separación
    "geo_dist_mean":   float,     # mean(geo_dist) de nodos fraude
    "time_s":          float,     # tiempo de entrenamiento en segundos
}
```

---

## Script de ejecución

```python
# diag73_run.py
import time, json, pickle
import numpy as np
import torch
from helix import HelixFramework
from helix.trainer import TrainConfig

# ── Cargar Elliptic ──────────────────────────────────────────────────────────
with open("data/funnel_elliptic.pkl", "rb") as f:
    data = pickle.load(f)

X          = data["X3"]          # (N, 20) — usar D=20 si disponible, D=3 si solo X3
ei         = data["edge_index"]  # (2, E)
labels_raw = data["labels"]      # -1=unknown, 0=licit, 1=illicit

# Convertir -1 a -1 (unlabeled), resto 0/1
labels = labels_raw.astype(np.float32)

BASELINE_AUC = 0.9624

RUNS = [
    # Grupo A
    {"run":"A1","spectral_norm":True,  "torque_lambda":0.00,"centroid_lambda":0.00,"directed":False,"auto_pca":None},
    {"run":"A2","spectral_norm":False, "torque_lambda":0.00,"centroid_lambda":0.00,"directed":False,"auto_pca":None},
    # Grupo B
    {"run":"B1","spectral_norm":True,  "torque_lambda":0.01,"centroid_lambda":0.00,"directed":False,"auto_pca":None},
    {"run":"B2","spectral_norm":True,  "torque_lambda":0.05,"centroid_lambda":0.00,"directed":False,"auto_pca":None},
    {"run":"B3","spectral_norm":True,  "torque_lambda":0.10,"centroid_lambda":0.00,"directed":False,"auto_pca":None},
    {"run":"B4","spectral_norm":True,  "torque_lambda":0.20,"centroid_lambda":0.00,"directed":False,"auto_pca":None},
    # Grupo C
    {"run":"C1","spectral_norm":True,  "torque_lambda":0.00,"centroid_lambda":0.05,"directed":False,"auto_pca":None},
    {"run":"C2","spectral_norm":True,  "torque_lambda":0.00,"centroid_lambda":0.10,"directed":False,"auto_pca":None},
    {"run":"C3","spectral_norm":True,  "torque_lambda":0.00,"centroid_lambda":0.20,"directed":False,"auto_pca":None},
    {"run":"C4","spectral_norm":True,  "torque_lambda":0.00,"centroid_lambda":0.50,"directed":False,"auto_pca":None},
    # Grupo D — rellenar best_* tras ver resultados B y C
    # {"run":"D1", ...},
    # {"run":"D2", ...},
    # {"run":"D3", ...},
]

results = []

for cfg_d in RUNS:
    print(f"\n{'='*50}")
    print(f"Run {cfg_d['run']} | torque={cfg_d['torque_lambda']} centroid={cfg_d['centroid_lambda']} sn={cfg_d['spectral_norm']}")

    cfg = TrainConfig(
        epochs=200, lr=1e-3, grad_clip=1.0, seed=42,
        torque_lambda=cfg_d["torque_lambda"],
        centroid_lambda=cfg_d["centroid_lambda"],
    )
    fw = HelixFramework(
        helix_hidden=16, helix_k=4, helix_t=5,
        spectral_norm=cfg_d["spectral_norm"],
        directed=cfg_d["directed"],
    )

    t0 = time.perf_counter()
    fw.fit(X, ei, labels, model="HELIX", cfg=cfg,
           auto_pca=cfg_d["auto_pca"])
    elapsed = time.perf_counter() - t0

    result = fw.explain(X, ei, labels=labels)

    row = {
        "run":            cfg_d["run"],
        "auc":            round(fw._helix_model._polarity_flipped and 0.0 or result.scores.max(), 4),  # placeholder
        "delta_vs_base":  None,
        "torque_mean":    float(result.torque.mean()),
        "geo_dist_mean":  float(result.geo_dist[labels==1].mean()) if (labels==1).any() else None,
        "time_s":         round(elapsed, 1),
    }
    # AUC real: necesita val_idx — simplificado aquí, en ejecución real usar train()
    results.append(row)
    print(json.dumps(row, indent=2))

# Guardar
with open("diag73_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDiag#73 completo. Ver diag73_results.json")
```

> **Nota:** El script anterior es esqueleto. Al ejecutar, reemplazar el cálculo de AUC
> por el valor de `TrainResult.auc` del loop de entrenamiento interno.

---

## Criterios de decisión post-experimento

| Resultado | Acción |
|-----------|--------|
| Ningún run supera +0.3pp | Mantener defaults actuales; los regularizadores no dañan pero no mejoran Elliptic |
| torque_λ* mejora >0.3pp | Fijar como default recomendado en docs; añadir a TrainConfig ejemplo |
| centroid_λ* mejora >0.3pp | Idem |
| D2 (directed) mejora | Documentar `directed=True` como recomendado para Elliptic específicamente |
| D3 (PCA+combo) mejora | Documentar `auto_pca=20` como recomendado para Elliptic |

---

## Ideas pendientes post-Diag#73

Las siguientes propuestas quedaron registradas como potencialmente interesantes
pero sin evidencia suficiente para implementar antes de validar los regularizadores actuales:

1. **Edge masking por coherencia de fase en phi_base**
   - Ponderar aristas según cos(θ) entre vectores phi_base de vecinos
   - Implementar como α aprendible en el Laplaciano antes del FNO
   - Evaluar solo si Diag#73 muestra que el cuello de botella sigue siendo la heterofilia

2. **Residual adaptativo por nodo (α_i)**
   - Reemplazar el α global de `residual_mix` por α_i = σ(W·‖phi_base_i‖)
   - Evaluar solo si los experimentos de Diag#73 muestran varianza alta entre nodos

**Condición de activación:** Si Diag#73 muestra que ningún regularizador ayuda y el
AUC plateau está en ~0.9624, entonces estas dos ideas pasan a Diag#74.

---

## Estado

- [ ] Grupo A ejecutado
- [ ] Grupo B ejecutado
- [ ] Grupo C ejecutado
- [ ] Grupo D configurado con best_* de B y C
- [ ] Grupo D ejecutado
- [ ] Resultados documentados en CHANGELOG
