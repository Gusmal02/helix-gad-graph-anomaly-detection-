# Helix

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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

fw = HelixFramework()
fw.fit(X, edge_index, labels)           # Graph Validator selects the best model

scores = fw.predict(X, edge_index)      # fast inference (Laser Query)
result = fw.explain(X, edge_index)      # full loop + geometric metrics
nexus  = fw.nexus(X, edge_index, confirmed=[12, 45, 891])   # S³ proximity scorer
sonar  = fw.sonar(X, edge_index, confirmed=[12, 45, 891])   # S³ + hop-distance scorer
report = fw.validate(X, edge_index, labels)                 # domain diagnostics

fw.save("model.pt")                     # persist to disk
fw2 = HelixFramework.load("model.pt")  # reload
```

Full example: [`examples/quick_start.py`](examples/quick_start.py)

---

## API

### `HelixFramework`

```python
HelixFramework(
    helix_hidden = 16,       # emission MLP hidden dimension
    helix_k      = 4,        # Chebyshev polynomial order
    helix_t      = 5,        # rotor loop steps
    gnn_hidden   = 64,       # hidden dimension for SAGE/GCN
    directed     = False,    # use random-walk Laplacian for directed graphs
)
```

#### `.fit(X, edge_index, labels, model='auto', cfg=None, edge_weight=None, auto_pca=None)`

Trains the framework. With `model='auto'`, the Graph Validator runs a quick probe (100 epochs × 3 seeds) and selects the optimal model. Always trains Helix internally — required for `explain()`, `nexus()`, and `sonar()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `ndarray (N, D)` | Node features |
| `edge_index` | `ndarray (2, E)` | Graph edges |
| `labels` | `ndarray (N,)` | Binary: 1=anomaly, 0=normal, -1=unlabeled |
| `model` | `str` | `'auto'` or force `'HELIX'` / `'SAGE'` / `'GCN'` / `'MLP'` |
| `cfg` | `TrainConfig` | Training hyperparameters |
| `edge_weight` | `ndarray (E,)` | Optional edge weights; uniform if `None` |
| `auto_pca` | `int \| None` | If set, reduce features to this many PCA components before training (e.g. `auto_pca=20`). Components are saved and applied automatically at inference time. |

#### `.predict(X, edge_index)` → `ndarray (N,)`

Fast inference via **Laser Query** (1 FNO step instead of T=5). Approximately 5× faster than the full loop with empirically equivalent AUC.

#### `.explain(X, edge_index)` → `ExplainResult`

Full Helix forward pass with geometric metrics. Always uses the Helix model for geometry, regardless of which model the Validator selected for scores.

```python
result.scores    # (N,) anomaly probabilities
result.rho       # (N,) ‖q_imag‖ — local instability
result.eta       # (N,) q_real — identity alignment
result.geo_dist  # (N,) arccos(|q_w|) — geodesic distance from identity
result.torque    # (N,) ‖τ‖ — torque magnitude (geometric gradient signal)
result.model_used   # model selected by the Validator
result.confidence   # 'HIGH' or 'MEDIUM'
result.q_final   # (N, 4) final quaternions in S³
```

> **Note:** If the domain has E/N > 10 or σ_seeds ≥ 0.025, `explain()` emits a `UserWarning` indicating that geometric metrics may be less reliable.

#### `.nexus(X, edge_index, confirmed, alpha=2.0)` → `ndarray (N,)`

Gravitational semi-supervised scorer. Propagates risk from confirmed anomalous seeds through S³ proximity without retraining.

```
nexus(j) = Σ_{i ∈ confirmed} exp(-α · arccos(|⟨q_i, q_j⟩|))
```

`alpha` controls the decay rate. Set `alpha='auto'` to calibrate from data as `1 / median_geo_dist(seeds, all_nodes)` — recommended when the typical S³ spread of the graph is unknown.

#### `.sonar(X, edge_index, confirmed, alpha=2.0, max_hops=4, hop_decay=0.6)` → `ndarray (N,)`

Extends NEXUS by combining S³ proximity with BFS hop distance. Nodes that are **both** geometrically close in S³ **and** few hops from a confirmed seed score highest — giving sharper propagation boundaries in sparse fraud clusters.

```
sonar(j) = nexus_normalized(j) · hop_decay^min_hops(j)
```

| Parameter | Description |
|-----------|-------------|
| `alpha` | Geodesic decay rate, or `'auto'` |
| `max_hops` | BFS depth limit; nodes beyond this receive `hop_decay^max_hops` penalty |
| `hop_decay` | Multiplier per hop (0 < hop_decay ≤ 1). Default 0.6 attenuates 40% per hop |

**When to use NEXUS vs SONAR:**
- Dense graph, many confirmed seeds → NEXUS (hop topology less informative)
- Sparse graph, few seeds, fraud clusters → SONAR (hop structure isolates clusters)
- Try both; SONAR subsumes NEXUS when `hop_decay=1.0`

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

#### `.save(path)` / `HelixFramework.load(path)`

Persist a trained framework to disk and reload it. Saves model weights, hyperparameters, PCA components (if `auto_pca` was used), and the validation report.

```python
fw.save("helix_elliptic.pt")
fw2 = HelixFramework.load("helix_elliptic.pt")
scores = fw2.predict(X_new, edge_index_new)
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
- **‖τ‖** — torque magnitude; high values indicate strong structural displacement

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

## NEXUS & SONAR — Semi-Supervised Mode

Useful when only a few confirmed fraud cases are available and the goal is to expand the investigation without retraining.

```python
confirmed = [42, 107, 891, 234, 56]

# NEXUS: S³ geodesic proximity only
risk_nexus = fw.nexus(X, edge_index, confirmed=confirmed, alpha='auto')

# SONAR: S³ proximity + hop distance (recommended for sparse fraud clusters)
risk_sonar = fw.sonar(X, edge_index, confirmed=confirmed,
                      alpha='auto', max_hops=4, hop_decay=0.6)

# Top-100 highest-risk nodes for investigation
candidates = np.argsort(risk_sonar)[-100:]
```

---

## TrainConfig

```python
from helix.trainer import TrainConfig

cfg = TrainConfig(
    epochs        = 200,    # training epochs
    lr            = 1e-3,   # Adam learning rate
    weight_decay  = 0.0,    # L2 regularization
    grad_clip     = 1.0,    # gradient norm clipping (prevents explosion in rotor loop)
    pos_weight    = None,   # positive class weight; auto-computed if None (= neg/pos ratio)
    seed          = 42,
    patience      = 0,      # early stopping patience (0 = disabled)
    min_delta     = 1e-4,   # minimum val-loss improvement to reset patience counter
    torque_lambda = 0.0,    # torque regularization weight λ·‖τ‖ (0 = disabled)
    verbose       = False,
    log_every     = 50,
)
```

**Early stopping:** set `patience > 0` to stop when validation loss stops improving. The best model state is restored at the end.

**Torque regularization:** `torque_lambda > 0` adds `λ · mean(‖τ‖)` to the loss, encouraging tighter geometric structure. Start with values in [0.01, 0.1].

---

## Directed Graphs

For transaction graphs where edge direction carries meaning (e.g. sender → receiver), use `directed=True`. This switches to a random-walk Laplacian (`D_out⁻¹ A`) instead of the symmetric Laplacian, which improves AUC on directed domains.

```python
fw = HelixFramework(directed=True)
fw.fit(X, edge_index, labels)
```

---

## Spectral Normalization

`ChebyshevFNO` wraps each per-channel filter with `torch.nn.utils.spectral_norm`, bounding its Lipschitz constant to ≤ 1 by dividing weights by their largest singular value on every forward pass. This prevents eigenvalue explosion in the Chebyshev coefficients and frequency collapse in domains where the graph topology is noisy or heterophilic — at zero runtime overhead.

Enabled by default. Disable only for ablation studies:

```python
fw = HelixFramework(spectral_norm=False)   # ablation only
```

---

## Tests

```bash
pytest                    # fast tests (67 tests, ~25s)
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
│   │   ├── laplacian.py   sparse Laplacian (symmetric + directed)
│   │   └── graph.py       structural diagnostics
│   ├── framework.py       HelixFramework — public API
│   ├── validator.py       Graph Validator
│   ├── trainer.py         unified training loop + early stopping
│   ├── nexus.py           NEXUS + SONAR scorers
│   └── metrics.py         ρ, η, F1_geo, σ_seeds, ratio_τ
├── tests/                 67 unit and integration tests
├── examples/
│   ├── quick_start.py     synthetic data demo
│   └── elliptic_demo.py   Bitcoin Elliptic demo
└── pyproject.toml
```
