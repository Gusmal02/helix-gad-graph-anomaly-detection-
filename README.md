# Helix

Framework de detección de anomalías en grafos que opera en el espacio de cuaterniones S³.

Diseñado para fraude financiero y dominios con grafos causales reales (transacciones, transferencias, llamadas). Incluye selección automática de modelo, métricas geométricas nativas y scoring semi-supervisado sin re-entrenamiento.

---

## Por qué Helix

Los modelos de grafos estándar (GCN, SAGE) tratan la detección de anomalías como clasificación tabular con propagación de mensajes. Helix toma un camino distinto: cada nodo recibe una **rotación** en S³ que refleja su posición estructural en el grafo. Los nodos ilícitos generan patrones geométricos distinguibles — torque, inestabilidad, desviación de la identidad — que el modelo aprende sin necesidad de features de fraude explícitas.

**Ley empírica validada en 6 dominios:** grafos causales reales con densidad E/N < 5 → Helix gana. Grafos kNN artificiales o muy densos → SAGE/MLP ganan. El `GraphValidator` aplica esta regla automáticamente.

---

## Benchmarks

| Dataset | Helix | SAGE | GCN | MLP | Ganador |
|---------|-------|------|-----|-----|---------|
| Elliptic (Bitcoin) | **0.9624** | 0.8833 | 0.8572 | 0.8902 | Helix |
| AMLSim (lavado) | **0.9647** | 0.8547 | 0.7927 | 0.9652* | Helix |
| PaySim (fraude móvil) | 0.9093 | **0.9768** | 0.9630 | — | SAGE |
| NF-UQ-NIDS (ciberseguridad) | 0.8120 | **0.9560** | 0.9523 | 0.7140 | SAGE |

*AMLSim MLP sin grafo (split por nodo). HELIX gana cuando el grafo es causal y sparse.

---

## Instalación

```bash
pip install -e .
```

Dependencias: `torch >= 2.0`, `numpy`, `scikit-learn`, `scipy`. Sin PyG ni DGL.

---

## Uso rápido

```python
import numpy as np
from helix import HelixFramework
from helix.trainer import TrainConfig

# Datos: features de nodos, grafo de transacciones, etiquetas binarias
fw = HelixFramework()
fw.fit(X, edge_index, labels)           # Graph Validator elige el mejor modelo

scores = fw.predict(X, edge_index)      # inferencia rápida (Laser Query)
result = fw.explain(X, edge_index)      # loop completo + métricas geométricas
nexus  = fw.nexus(X, edge_index, confirmed=[12, 45, 891])  # propagación semi-supervisada
report = fw.validate(X, edge_index, labels)                # diagnóstico del dominio
```

Ejemplo completo: [`examples/quick_start.py`](examples/quick_start.py)

---

## API

### `HelixFramework`

```python
HelixFramework(
    helix_hidden = 16,   # dimensión del MLP de emisión
    helix_k      = 4,    # orden Chebyshev
    helix_t      = 5,    # pasos del loop de rotores
    gnn_hidden   = 64,   # dimensión para SAGE/GCN
)
```

#### `.fit(X, edge_index, labels, model='auto', cfg=None)`

Entrena el framework. Con `model='auto'`, el Graph Validator corre un probe rápido (100 épocas × 3 seeds) y selecciona el modelo óptimo. Siempre entrena Helix internamente — necesario para `explain()` y `nexus()`.

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `X` | `ndarray (N, D)` | Features por nodo |
| `edge_index` | `ndarray (2, E)` | Aristas del grafo |
| `labels` | `ndarray (N,)` | Binario: 1=anomalía, 0=normal, -1=sin etiqueta |
| `model` | `str` | `'auto'` / `'HELIX'` / `'SAGE'` / `'GCN'` / `'MLP'` |
| `cfg` | `TrainConfig` | Hiperparámetros de entrenamiento |

#### `.predict(X, edge_index)` → `ndarray (N,)`

Inferencia rápida vía **Laser Query** (1 paso FNO en lugar de T=5). Aproximadamente 5× más rápido que el loop completo con AUC equivalente empíricamente.

#### `.explain(X, edge_index)` → `ExplainResult`

Loop completo de Helix con métricas geométricas. Siempre usa el modelo Helix para la geometría, independientemente de qué modelo ganó en el Validator.

```python
result.scores    # (N,) probabilidades de anomalía
result.rho       # (N,) norma imaginaria — inestabilidad local
result.eta       # (N,) parte real — alineación con identidad
result.geo_dist  # (N,) distancia geodésica desde quaternión identidad
result.model_used   # modelo seleccionado por el Validator
result.confidence   # 'ALTA' o 'MEDIA'
result.q_final   # (N, 4) quaterniones finales en S³
```

> **Nota:** Si el dominio tiene E/N > 10 o σ_seeds ≥ 0.025, `explain()` emite un `UserWarning` indicando que las métricas geométricas pueden ser menos confiables.

#### `.nexus(X, edge_index, confirmed, alpha=2.0)` → `ndarray (N,)`

Scoring gravitatorio semi-supervisado. Dados nodos confirmados como anómalos, propaga una puntuación de riesgo por proximidad en S³ sin re-entrenar.

```
nexus(j) = Σ_{i ∈ confirmed} exp(-α · arccos(|⟨q_i, q_j⟩|))
```

`alpha` controla la velocidad de decaimiento: valores altos concentran el score cerca de los confirmados.

#### `.validate(X, edge_index, labels)` → `ValidationReport`

Diagnóstico estructural del grafo + probe rápido. Devuelve:

```python
report.recommended_model   # 'HELIX', 'SAGE', 'GCN' o 'MLP'
report.confidence          # 'ALTA' o 'MEDIA'
report.reason              # explicación de la regla aplicada
report.sigma_seeds         # σ inter-seeds del probe Helix
report.graph_density       # E/N
report.gini                # coeficiente de Gini del grado
report.homophily           # fracción de aristas entre nodos del mismo label
report.helix_auc_mean      # AUC media del probe
report.lift                # HELIX AUC − MLP AUC
```

---

## Arquitectura del modelo Helix

```
X (N×D)
  │
  ▼
EmissionMLP(D → hidden → 3)          features → campo vectorial 3D
  │
  ▼
ChebyshevFNO(K=4, sparse)            propagación espectral del campo
  │
  ├─→ τ = φ_base × Φ_prop           torque: campo base × campo propagado
  │
  ▼
Rotor loop × T=5                     actualización en S³
  q_{t+1} = exp(η·‖τ‖/2 · τ̂) ⊗ exp(ω·dt/2 · â) ⊗ q_t
  │
  ▼
v_final = q_T · φ_base · q_T*       rotación del campo base
  │
  ▼
Linear(3 → 1)                        logit de clasificación
```

**Métricas nativas** (sin post-hoc):
- **ρ = ‖q_imag‖** — inestabilidad local del nodo
- **η = q_real** — alineación con el rotor identidad
- **dist_geo = arccos(|q_w|)** — desviación en S³

---

## Graph Validator — reglas de decisión

El Validator aplica 4 reglas en orden de prioridad, calibradas en 4 dominios reales:

| Regla | Condición | Modelo | Confianza |
|-------|-----------|--------|-----------|
| 1 | E/N > 10 (grafo denso) | SAGE | ALTA |
| 2 | Gini > 0.6 AND homophily > 0.95 (hub-and-spoke) | SAGE | ALTA |
| 3 | σ_seeds ≥ 0.025 AND lift < −0.05 (inestable, MLP gana) | MLP | ALTA |
| 3b | σ_seeds ≥ 0.025 (inestable) | SAGE | MEDIA |
| 4 | σ_seeds < 0.025 AND E/N ≤ 10 | HELIX | ALTA/MEDIA |

Precisión empírica: **4/4 dominios** (Elliptic, AMLSim, PaySim, Electricity).

---

## NEXUS — modo semi-supervisado

Útil cuando se tienen pocos casos confirmados de fraude y se quiere expandir la investigación sin re-entrenar. Helix mapea cada nodo a un punto en S³; NEXUS mide qué tan cerca está cada nodo de los confirmados en ese espacio.

```python
# 5 cuentas confirmadas como fraudulentas por operaciones
confirmed = [42, 107, 891, 234, 56]
risk = fw.nexus(X, edge_index, confirmed=confirmed, alpha=2.0)

# Top-100 nodos de mayor riesgo para investigación
candidates = np.argsort(risk)[-100:]
```

---

## TrainConfig

```python
from helix.trainer import TrainConfig

cfg = TrainConfig(
    epochs       = 200,    # épocas de entrenamiento
    lr           = 1e-3,   # learning rate Adam
    weight_decay = 0.0,    # regularización L2
    grad_clip    = 1.0,    # clip de gradientes (previene explosión en rotores)
    pos_weight   = None,   # peso clase positiva; auto-calculado si None
    seed         = 42,
    verbose      = False,
    log_every    = 50,
)
```

---

## Tests

```bash
pytest                    # tests rápidos (44 tests, ~6s)
pytest -m slow            # incluye probe completo del Validator (~60s)
```

---

## Estructura

```
helix/
├── helix/
│   ├── models/
│   │   ├── helix.py       modelo principal + laser_query
│   │   ├── sage.py        GraphSAGE (sin PyG)
│   │   ├── gcn.py         GCN (sin PyG)
│   │   └── mlp.py         baseline tabular
│   ├── core/
│   │   ├── rotors.py      aritmética cuaterniónica en S³
│   │   ├── chebyshev.py   propagador espectral Chebyshev
│   │   ├── laplacian.py   Laplaciana normalizada sparse
│   │   └── graph.py       diagnósticos estructurales
│   ├── framework.py       HelixFramework — API pública
│   ├── validator.py       Graph Validator
│   ├── trainer.py         loop de entrenamiento unificado
│   ├── nexus.py           scorer gravitatorio NEXUS
│   └── metrics.py         ρ, η, F1_geo, σ_seeds, ratio_τ
├── tests/                 44 tests unitarios e integración
├── examples/
│   ├── quick_start.py     demo en datos sintéticos
│   └── elliptic_demo.py   demo con Bitcoin Elliptic
└── pyproject.toml
```
