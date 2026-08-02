# API Reference — Helix

---

## `helix.HelixFramework`

Main class. Orchestrates model selection, training, inference, and metrics.

```python
from helix import HelixFramework
```

### Constructor

```python
HelixFramework(
    helix_hidden: int = 16,
    helix_k: int = 4,
    helix_t: int = 5,
    gnn_hidden: int = 64,
)
```

| Parameter | Description |
|-----------|-------------|
| `helix_hidden` | Hidden dimension of the Helix emission MLP |
| `helix_k` | Chebyshev polynomial order (spectral depth) |
| `helix_t` | Number of rotor loop steps |
| `gnn_hidden` | Hidden dimension for SAGE and GCN when selected by the Validator |

---

### `.fit(X, edge_index, labels, model='auto', cfg=None, edge_weight=None)`

Trains the framework. Returns `self` for chaining.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `X` | `np.ndarray (N, D)` | Node features, float32 |
| `edge_index` | `np.ndarray (2, E)` | Graph edges, int64 |
| `labels` | `np.ndarray (N,)` | 1=anomaly, 0=normal, -1=unlabeled |
| `model` | `str` | `'auto'` lets the Validator choose; or force `'HELIX'` / `'SAGE'` / `'GCN'` / `'MLP'` |
| `cfg` | `TrainConfig` | Training configuration (see TrainConfig section) |
| `edge_weight` | `np.ndarray (E,)` | Edge weights; uniform if `None` |

**Internal behavior:**
- Always trains the Helix model (required by `explain()` and `nexus()`).
- If the Validator selects a different model, it is trained additionally.
- `predict()` uses the winning model; `explain()` and `nexus()` always use Helix.

---

### `.predict(X, edge_index, edge_weight=None)` → `np.ndarray (N,)`

Fast inference. Uses **Laser Query** when the active model is Helix (1 FNO step instead of T=5 rotor steps), or the standard forward pass for SAGE/GCN/MLP.

Returns scores in `[0, 1]` via sigmoid.

---

### `.explain(X, edge_index, edge_weight=None, labels=None)` → `ExplainResult`

Full Helix forward pass with geometric metric computation.

**Returns `ExplainResult` with fields:**

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `ndarray (N,)` | Anomaly probabilities from the winning model |
| `rho` | `ndarray (N,)` | ρ = ‖q_imag‖ — local instability |
| `eta` | `ndarray (N,)` | η = q_real — identity alignment |
| `geo_dist` | `ndarray (N,)` | arccos(\|q_w\|) — geodesic distance in S³ |
| `model_used` | `str` | Model that generated `scores` |
| `confidence` | `str` | `'HIGH'` or `'MEDIUM'` from the Validator (`None` if model was forced) |
| `q_final` | `ndarray (N, 4)` | Final quaternions in S³ |

**Warnings emitted:**
- `UserWarning` if E/N > 10: geometric metrics are less reliable on dense graphs.
- `UserWarning` if σ_seeds ≥ 0.025: Helix is unstable on this domain.

---

### `.nexus(X, edge_index, confirmed, alpha=2.0, edge_weight=None)` → `np.ndarray (N,)`

Gravitational semi-supervised scorer. Does not retrain.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `confirmed` | `list[int]` | Indices of nodes confirmed as anomalous |
| `alpha` | `float` | Geodesic decay rate. Higher = scores more concentrated near confirmed nodes |

**Formula:**
```
nexus(j) = Σ_{i ∈ confirmed} exp(-α · arccos(|⟨q_i, q_j⟩|)) / |confirmed|
```

Returns scores in `[0, 1]`.

---

### `.validate(X, edge_index, labels)` → `ValidationReport`

Structural diagnostics + quick probe. Does not modify framework state.

**Returns `ValidationReport` with fields:**

| Field | Type | Description |
|-------|------|-------------|
| `recommended_model` | `str` | `'HELIX'`, `'SAGE'`, `'GCN'`, or `'MLP'` |
| `confidence` | `str` | `'HIGH'` or `'MEDIUM'` |
| `reason` | `str` | Rule that triggered the decision with numeric values |
| `sigma_seeds` | `float` | Inter-seed AUC std of the probe (3 seeds × 100 epochs) |
| `graph_density` | `float` | E/N — edges per node |
| `gini` | `float` | Degree Gini coefficient. > 0.6 indicates hub-and-spoke |
| `homophily` | `float` | Fraction of edges between same-label nodes |
| `helix_auc_mean` | `float` | Mean Helix AUC from the probe |
| `helix_auc_std` | `float` | AUC std (= `sigma_seeds`) |
| `mlp_auc` | `float` | Reference MLP AUC |
| `lift` | `float` | `helix_auc_mean - mlp_auc` |

---

## `helix.trainer.TrainConfig`

```python
from helix.trainer import TrainConfig
```

| Field | Default | Description |
|-------|---------|-------------|
| `epochs` | `200` | Training epochs |
| `lr` | `1e-3` | Learning rate (Adam) |
| `weight_decay` | `0.0` | L2 regularization |
| `grad_clip` | `1.0` | Gradient norm clipping. Prevents explosion in the rotor loop |
| `pos_weight` | `None` | Positive class weight in `BCEWithLogitsLoss`. Auto-computed if `None` (= neg/pos ratio) |
| `seed` | `42` | Reproducibility seed |
| `verbose` | `False` | Logs loss every `log_every` epochs |
| `log_every` | `50` | Logging frequency when `verbose=True` |

---

## `helix.metrics`

Geometric metric functions. Operate directly on `q_final (N, 4)`.

```python
from helix.metrics import rho, eta, geo_distance, geo_flag, best_f1_geo, sigma_seeds
```

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `rho(q)` | `(N,4)` | `(N,)` | ρ = ‖q_imag‖ |
| `eta(q)` | `(N,4)` | `(N,)` | η = q_real |
| `geo_distance(q)` | `(N,4)` | `(N,)` | arccos(\|q_w\|) — distance to identity in S³ |
| `geo_flag(q, theta)` | `(N,4), float` | `(N,)` bool | `True` where dist > theta |
| `best_f1_geo(q, labels)` | `(N,4), (N,)` | `(float, float)` | Best F1_geo and optimal θ |
| `sigma_seeds(auc_list)` | `list[float]` | `float` | Inter-seed standard deviation |

---

## `helix.nexus`

```python
from helix.nexus import nexus_score, nexus_score_normalized
```

| Function | Description |
|----------|-------------|
| `nexus_score(q_final, confirmed_idx, alpha)` | Raw score — range `[0, len(confirmed)]` |
| `nexus_score_normalized(q_final, confirmed_idx, alpha)` | Normalized score — range `[0, 1]` |

---

## `helix.models`

Models can be used directly without the framework.

```python
from helix.models.helix import HelixModel
from helix.models.sage  import SAGEModel
from helix.models.gcn   import GCNModel
from helix.models.mlp   import MLPModel
```

All share the same `forward` signature:

```python
model(x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None)
```

- **HelixModel** returns `(logits (N,1), q_final (N,4))`
- All others return `logits (N,1)`

`HelixModel` also exposes `.laser_query(x, edge_index, edge_weight)` → `logits (N,1)`.

---

## `helix.core`

Reusable primitives.

```python
from helix.core.rotors    import quat_normalize, quat_exp, quat_mul, quat_rotate, RotorStep
from helix.core.chebyshev import ChebyshevFNO, chebyshev_propagate
from helix.core.laplacian import normalized_laplacian
from helix.core.graph     import edge_density, degree_gini, label_homophily, graph_stats
```

| Function / Class | Description |
|-----------------|-------------|
| `quat_normalize(q)` | Normalizes to ‖q‖=1 |
| `quat_exp(v)` | Quaternion exponential of pure vector v∈R³ → q∈S³ |
| `quat_mul(q1, q2)` | Hamilton product |
| `quat_rotate(q, v)` | Rotation v' = q v q* |
| `RotorStep` | nn.Module — one rotor loop step (no .detach()) |
| `ChebyshevFNO` | Chebyshev spectral operator over a 3D vector field |
| `normalized_laplacian(edge_index, edge_weight, N)` | Returns (L_tilde, lambda_max). Supports directed graphs |
| `edge_density(ei, N)` | E/N |
| `degree_gini(ei, N)` | Degree Gini coefficient |
| `label_homophily(ei, labels)` | Fraction of homophilic edges |
| `graph_stats(ei, N, labels)` | Dict with density, gini, homophily |
