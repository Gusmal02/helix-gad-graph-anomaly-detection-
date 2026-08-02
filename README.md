# Helix

Graph anomaly detection framework operating in the unit quaternion sphere S³.

Designed for financial fraud and domains with real causal graphs (transactions, transfers, network flows). Features automatic model selection, native geometric interpretability metrics, and semi-supervised scoring without retraining.

---

## Why Helix

Standard graph models (GCN, SAGE) treat anomaly detection as tabular classification with message passing. Helix takes a different path: each node receives a **rotation** in S³ that reflects its structural position in the graph. Illicit nodes generate distinguishable geometric patterns — torque, instability, deviation from identity — that the model learns without requiring explicit fraud features.

**Empirical law validated across 6 domains:** real causal graphs with density E/N < 5 → Helix wins. Artificial kNN graphs or dense graphs → SAGE/MLP win. The `GraphValidator` applies this rule automatically.

---

## Benchmarks

| Dataset | Helix | SAGE | GCN | MLP | Winner |
|---------|-------|------|-----|-----|--------|
| Elliptic (Bitcoin) | **0.9624** | 0.8833 | 0.8572 | 0.8902 | Helix |
| AMLSim (money laundering) | **0.9647** | 0.8547 | 0.7927 | 0.9652* | Helix |
| PaySim (mobile fraud) | 0.9093 | **0.9768** | 0.9630 | — | SAGE |
| NF-UQ-NIDS (cybersecurity) | 0.8120 | **0.9560** | 0.9523 | 0.7140 | SAGE |

*AMLSim MLP without graph (node-level split). Helix wins when the graph is causal and sparse.

---

## Installation

```bash
pip install -e .
```

Dependencies: `torch >= 2.0`, `numpy`, `scikit-learn`, `scipy`. No PyG or DGL required.

---

## Quick Start

```python
import numpy as np
from helix import HelixFramework
from helix.trainer import TrainConfig

# Data: node features, transaction graph, binary labels
fw = HelixFramework()
fw.fit(X, edge_index, labels)           # Graph Validator selects the best model

scores = fw.predict(X, edge_index)      # fast inference (Laser Query)
result = fw.explain(X, edge_index)      # full loop + geometric metrics
nexus  = fw.nexus(X, edge_index, confirmed=[12, 45, 891])  # semi-supervised propagation
report = fw.validate(X, edge_index, labels)                # domain diagnostics
```

Full example: [`examples/quick_start.py`](examples/quick_start.py)

---

## API

### `HelixFramework`

```python
HelixFramework(
    helix_hidden = 16,   # emission MLP hidden dimension
    helix_k      = 4,    # Chebyshev polynomial order
    helix_t      = 5,    # rotor loop steps
    gnn_hidden   = 64,   # hidden dimension for SAGE/GCN
)
```

#### `.fit(X, edge_index, labels, model='auto', cfg=None)`

Trains the framework. With `model='auto'`, the Graph Validator runs a quick probe (100 epochs × 3 seeds) and selects the optimal model. Always trains Helix internally — required for `explain()` and `nexus()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `ndarray (N, D)` | Node features |
| `edge_index` | `ndarray (2, E)` | Graph edges |
| `labels` | `ndarray (N,)` | Binary: 1=anomaly, 0=normal, -1=unlabeled |
| `model` | `str` | `'auto'` or force `'HELIX'` / `'SAGE'` / `'GCN'` / `'MLP'` |
| `cfg` | `TrainConfig` | Training hyperparameters |

#### `.predict(X, edge_index)` → `ndarray (N,)`

Fast inference via **Laser Query** (1 FNO step instead of T=5). Approximately 5× faster than the full loop with empirically equivalent AUC.

#### `.explain(X, edge_index)` → `ExplainResult`

Full Helix forward pass with geometric metrics. Always uses the Helix model for geometry, regardless of which model the Validator selected for scores.

```python
result.scores    # (N,) anomaly probabilities
result.rho       # (N,) imaginary norm — local instability
result.eta       # (N,) scalar part — identity alignment
result.geo_dist  # (N,) geodesic distance from identity quaternion
result.model_used   # model selected by the Validator
result.confidence   # 'HIGH' or 'MEDIUM'
result.q_final   # (N, 4) final quaternions in S³
```

> **Note:** If the domain has E/N > 10 or σ_seeds ≥ 0.025, `explain()` emits a `UserWarning` indicating that geometric metrics may be less reliable.

#### `.nexus(X, edge_index, confirmed, alpha=2.0)` → `ndarray (N,)`

Gravitational semi-supervised scorer. Given confirmed anomalous nodes, propagates a risk score through S³ proximity without retraining.

```
nexus(j) = Σ_{i ∈ confirmed} exp(-α · arccos(|⟨q_i, q_j⟩|))
```

`alpha` controls the decay rate: higher values concentrate scores near confirmed nodes.

#### `.validate(X, edge_index, labels)` → `ValidationReport`

Structural graph diagnostics + quick probe. Returns:

```python
report.recommended_model   # 'HELIX', 'SAGE', 'GCN', or 'MLP'
report.confidence          # 'HIGH' or 'MEDIUM'
report.reason              # explanation of the applied rule with numeric values
report.sigma_seeds         # inter-seed AUC std of the Helix probe
report.graph_density       # E/N
report.gini                # degree Gini coefficient
report.homophily           # fraction of edges between same-label nodes
report.helix_auc_mean      # mean Helix AUC from the probe
report.lift                # HELIX AUC − MLP AUC
```

---

## Helix Model Architecture

```
X (N×D)
  │
  ▼
EmissionMLP(D → hidden → 3)          features → 3D vector field
  │
  ▼
ChebyshevFNO(K=4, sparse)            spectral field propagation
  │
  ├─→ τ = φ_base × Φ_prop           torque: base field × propagated field
  │
  ▼
Rotor loop × T=5                     update in S³
  q_{t+1} = exp(η·‖τ‖/2 · τ̂) ⊗ exp(ω·dt/2 · â) ⊗ q_t
  │
  ▼
v_final = q_T · φ_base · q_T*       rotate base field by final quaternion
  │
  ▼
Linear(3 → 1)                        classification logit
```

**Native metrics** (no post-hoc explanation needed):
- **ρ = ‖q_imag‖** — local node instability
- **η = q_real** — alignment with identity rotor
- **dist_geo = arccos(|q_w|)** — deviation in S³

---

## Graph Validator — Decision Rules

The Validator applies 4 rules in priority order, calibrated on 4 real fraud domains:

| Rule | Condition | Model | Confidence |
|------|-----------|-------|------------|
| 1 | E/N > 10 (dense graph) | SAGE | HIGH |
| 2 | Gini > 0.6 AND homophily > 0.95 (hub-and-spoke) | SAGE | HIGH |
| 3 | σ_seeds ≥ 0.025 AND lift < −0.05 (unstable, MLP wins) | MLP | HIGH |
| 3b | σ_seeds ≥ 0.025 (unstable) | SAGE | MEDIUM |
| 4 | σ_seeds < 0.025 AND E/N ≤ 10 | HELIX | HIGH/MEDIUM |

Empirical accuracy: **4/4 domains** (Elliptic, AMLSim, PaySim, Electricity).

---

## NEXUS — Semi-Supervised Mode

Useful when only a few confirmed fraud cases are available and the goal is to expand the investigation without retraining. Helix maps each node to a point in S³; NEXUS measures how close each node is to the confirmed cases in that space.

```python
# 5 accounts confirmed as fraudulent by operations
confirmed = [42, 107, 891, 234, 56]
risk = fw.nexus(X, edge_index, confirmed=confirmed, alpha=2.0)

# Top-100 highest-risk nodes for investigation
candidates = np.argsort(risk)[-100:]
```

---

## TrainConfig

```python
from helix.trainer import TrainConfig

cfg = TrainConfig(
    epochs       = 200,    # training epochs
    lr           = 1e-3,   # Adam learning rate
    weight_decay = 0.0,    # L2 regularization
    grad_clip    = 1.0,    # gradient norm clipping (prevents explosion in rotor loop)
    pos_weight   = None,   # positive class weight; auto-computed if None (= neg/pos ratio)
    seed         = 42,
    verbose      = False,
    log_every    = 50,
)
```

---

## Tests

```bash
pytest                    # fast tests (44 tests, ~6s)
pytest -m slow            # includes full Validator probe (~60s)
```

---

## Repository Structure

```
helix/
├── helix/
│   ├── models/
│   │   ├── helix.py       main model + laser_query
│   │   ├── sage.py        GraphSAGE (no PyG)
│   │   ├── gcn.py         GCN (no PyG)
│   │   └── mlp.py         tabular baseline
│   ├── core/
│   │   ├── rotors.py      quaternion arithmetic in S³
│   │   ├── chebyshev.py   Chebyshev spectral propagator
│   │   ├── laplacian.py   sparse normalized Laplacian
│   │   └── graph.py       structural diagnostics
│   ├── framework.py       HelixFramework — public API
│   ├── validator.py       Graph Validator
│   ├── trainer.py         unified training loop
│   ├── nexus.py           NEXUS gravitational scorer
│   └── metrics.py         ρ, η, F1_geo, σ_seeds, ratio_τ
├── tests/                 44 unit and integration tests
├── examples/
│   ├── quick_start.py     synthetic data demo
│   └── elliptic_demo.py   Bitcoin Elliptic demo
└── pyproject.toml
```
