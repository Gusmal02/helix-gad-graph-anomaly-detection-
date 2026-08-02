# Referencia de API — Helix

---

## `helix.HelixFramework`

Clase principal. Orquesta selección de modelo, entrenamiento, inferencia y métricas.

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

| Parámetro | Descripción |
|-----------|-------------|
| `helix_hidden` | Dimensión de la capa oculta del MLP de emisión en Helix |
| `helix_k` | Orden K del polinomio de Chebyshev (profundidad espectral) |
| `helix_t` | Número de pasos del loop de rotores |
| `gnn_hidden` | Dimensión oculta para SAGE y GCN cuando el Validator los selecciona |

---

### `.fit(X, edge_index, labels, model='auto', cfg=None, edge_weight=None)`

Entrena el framework. Retorna `self` para encadenamiento.

**Parámetros:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `X` | `np.ndarray (N, D)` | Features de nodos, float32 |
| `edge_index` | `np.ndarray (2, E)` | Aristas del grafo, int64 |
| `labels` | `np.ndarray (N,)` | 1=anomalía, 0=normal, -1=sin etiqueta |
| `model` | `str` | `'auto'` deja que el Validator elija; o fuerza `'HELIX'` / `'SAGE'` / `'GCN'` / `'MLP'` |
| `cfg` | `TrainConfig` | Configuración de entrenamiento (ver sección TrainConfig) |
| `edge_weight` | `np.ndarray (E,)` | Pesos de aristas; uniforme si `None` |

**Comportamiento interno:**
- Siempre entrena el modelo Helix (requerido por `explain()` y `nexus()`).
- Si el Validator elige un modelo distinto a Helix, lo entrena adicionalmente.
- `predict()` usa el modelo ganador; `explain()` y `nexus()` siempre usan Helix.

---

### `.predict(X, edge_index, edge_weight=None)` → `np.ndarray (N,)`

Inferencia rápida. Usa **Laser Query** cuando el modelo activo es Helix (1 paso FNO en lugar de T=5 pasos), o el forward estándar para SAGE/GCN/MLP.

Retorna scores en `[0, 1]` vía sigmoid.

---

### `.explain(X, edge_index, edge_weight=None, labels=None)` → `ExplainResult`

Loop completo de Helix con cálculo de métricas geométricas.

**Retorna `ExplainResult` con campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `scores` | `ndarray (N,)` | Probabilidades de anomalía del modelo ganador |
| `rho` | `ndarray (N,)` | ρ = ‖q_imag‖ — inestabilidad local |
| `eta` | `ndarray (N,)` | η = q_real — alineación con identidad |
| `geo_dist` | `ndarray (N,)` | arccos(\|q_w\|) — distancia geodésica en S³ |
| `model_used` | `str` | Modelo que generó `scores` |
| `confidence` | `str` | `'ALTA'` o `'MEDIA'` del Validator (`None` si `model` fue forzado) |
| `q_final` | `ndarray (N, 4)` | Quaterniones finales en S³ |

**Advertencias emitidas:**
- `UserWarning` si E/N > 10: métricas geométricas menos confiables en grafos densos.
- `UserWarning` si σ_seeds ≥ 0.025: Helix inestable en este dominio.

---

### `.nexus(X, edge_index, confirmed, alpha=2.0, edge_weight=None)` → `np.ndarray (N,)`

Scorer gravitatorio semi-supervisado. No re-entrena.

**Parámetros:**

| Nombre | Tipo | Descripción |
|--------|------|-------------|
| `confirmed` | `list[int]` | Índices de nodos confirmados como anómalos |
| `alpha` | `float` | Tasa de decaimiento geodésico. Más alto = score más concentrado cerca de confirmados |

**Fórmula:**
```
nexus(j) = Σ_{i ∈ confirmed} exp(-α · arccos(|⟨q_i, q_j⟩|)) / |confirmed|
```

Retorna scores en `[0, 1]`.

---

### `.validate(X, edge_index, labels)` → `ValidationReport`

Diagnóstico estructural + probe rápido. No modifica el estado del framework.

**Retorna `ValidationReport` con campos:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `recommended_model` | `str` | `'HELIX'`, `'SAGE'`, `'GCN'` o `'MLP'` |
| `confidence` | `str` | `'ALTA'` o `'MEDIA'` |
| `reason` | `str` | Regla que activó la decisión y valores numéricos |
| `sigma_seeds` | `float` | σ inter-seeds del probe (3 seeds × 100 épocas) |
| `graph_density` | `float` | E/N — aristas por nodo |
| `gini` | `float` | Coeficiente de Gini del grado. > 0.6 indica hub-and-spoke |
| `homophily` | `float` | Fracción de aristas entre nodos del mismo label |
| `helix_auc_mean` | `float` | AUC media de Helix en el probe |
| `helix_auc_std` | `float` | σ de AUC (= `sigma_seeds`) |
| `mlp_auc` | `float` | AUC del MLP de referencia |
| `lift` | `float` | `helix_auc_mean - mlp_auc` |

---

## `helix.trainer.TrainConfig`

```python
from helix.trainer import TrainConfig
```

| Campo | Por defecto | Descripción |
|-------|-------------|-------------|
| `epochs` | `200` | Épocas de entrenamiento |
| `lr` | `1e-3` | Learning rate (Adam) |
| `weight_decay` | `0.0` | Regularización L2 |
| `grad_clip` | `1.0` | Clip de norma del gradiente. Previene explosión en el loop de rotores |
| `pos_weight` | `None` | Peso de la clase positiva en `BCEWithLogitsLoss`. Auto-calculado si `None` (= neg/pos) |
| `seed` | `42` | Semilla para reproducibilidad |
| `verbose` | `False` | Imprime loss cada `log_every` épocas |
| `log_every` | `50` | Frecuencia de logging si `verbose=True` |

---

## `helix.metrics`

Funciones de métricas geométricas. Operan directamente sobre `q_final (N, 4)`.

```python
from helix.metrics import rho, eta, geo_distance, geo_flag, best_f1_geo, sigma_seeds
```

| Función | Entrada | Salida | Descripción |
|---------|---------|--------|-------------|
| `rho(q)` | `(N,4)` | `(N,)` | ρ = ‖q_imag‖ |
| `eta(q)` | `(N,4)` | `(N,)` | η = q_real |
| `geo_distance(q)` | `(N,4)` | `(N,)` | arccos(\|q_w\|) — distancia a identidad en S³ |
| `geo_flag(q, theta)` | `(N,4), float` | `(N,)` bool | `True` donde dist > theta |
| `best_f1_geo(q, labels)` | `(N,4), (N,)` | `(float, float)` | Mejor F1_geo y θ óptimo |
| `sigma_seeds(auc_list)` | `list[float]` | `float` | Desviación estándar inter-seeds |

---

## `helix.nexus`

```python
from helix.nexus import nexus_score, nexus_score_normalized
```

| Función | Descripción |
|---------|-------------|
| `nexus_score(q_final, confirmed_idx, alpha)` | Score crudo — rango `[0, len(confirmed)]` |
| `nexus_score_normalized(q_final, confirmed_idx, alpha)` | Score normalizado — rango `[0, 1]` |

---

## `helix.models`

Los modelos se pueden usar directamente sin el framework.

```python
from helix.models.helix import HelixModel
from helix.models.sage  import SAGEModel
from helix.models.gcn   import GCNModel
from helix.models.mlp   import MLPModel
```

Todos comparten la misma firma de `forward`:

```python
model(x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None)
```

- **HelixModel** retorna `(logits (N,1), q_final (N,4))`
- El resto retorna `logits (N,1)`

`HelixModel` incluye además `.laser_query(x, edge_index, edge_weight)` → `logits (N,1)`.

---

## `helix.core`

Primitivas reutilizables.

```python
from helix.core.rotors    import quat_normalize, quat_exp, quat_mul, quat_rotate, RotorStep
from helix.core.chebyshev import ChebyshevFNO, chebyshev_propagate
from helix.core.laplacian import normalized_laplacian
from helix.core.graph     import edge_density, degree_gini, label_homophily, graph_stats
```

| Función / Clase | Descripción |
|----------------|-------------|
| `quat_normalize(q)` | Normaliza a ‖q‖=1 |
| `quat_exp(v)` | Exponencial cuaterniónica de vector puro v∈R³ → q∈S³ |
| `quat_mul(q1, q2)` | Producto de Hamilton |
| `quat_rotate(q, v)` | Rotación v'= q v q* |
| `RotorStep` | Módulo nn — un paso del loop de rotores (sin .detach()) |
| `ChebyshevFNO` | Operador espectral Chebyshev sobre campo 3D |
| `normalized_laplacian(edge_index, edge_weight, N)` | Retorna (L_tilde, lambda_max). Soporta grafos dirigidos |
| `edge_density(ei, N)` | E/N |
| `degree_gini(ei, N)` | Coeficiente de Gini del grado |
| `label_homophily(ei, labels)` | Fracción de aristas homofílicas |
| `graph_stats(ei, N, labels)` | Dict con density, gini, homophily |
