# Helix — Phase 1 Plan
**Directorio:** `C:\Users\Gustavo\Documents\tripleten\proyectos\helix`
**Fecha:** 2026-08-01
**Estado:** Planeación

---

## Qué es Helix

Framework de detección de anomalías en grafos con tres modos de operación:

- **predict()** — inferencia rápida vía Laser Query (5× más rápido, mismo AUC)
- **explain()** — loop completo con métricas geométricas nativas (ρ, τ, F1_geo)
- **nexus()** — scoring gravitatorio semi-supervisado desde casos confirmados

El núcleo es el modelo HELIX (cuaterniones en S³). El framework incluye SAGE, GCN y MLP como alternativas, con un Graph Validator que elige automáticamente el modelo correcto según el dominio.

---

## Estructura del paquete

```
helix/
├── helix/                        # paquete principal
│   ├── __init__.py               # exports públicos
│   ├── models/
│   │   ├── __init__.py
│   │   ├── helix.py              # modelo HELIX (cuaterniones)
│   │   ├── sage.py               # GraphSAGE
│   │   ├── gcn.py                # GCN
│   │   └── mlp.py                # MLP
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chebyshev.py          # Chebyshev FNO (portado de gfcn/chebyshev.py)
│   │   ├── rotors.py             # operaciones cuaterniónicas (portado de gfcn/rotors.py)
│   │   ├── laplacian.py          # construcción de L_tilde
│   │   └── graph.py              # utilidades de grafo (degree, homophily, gini)
│   ├── validator.py              # Graph Validator — elige HELIX/SAGE/MLP
│   ├── trainer.py                # loop de entrenamiento unificado
│   ├── metrics.py                # ρ, τ, F1_geo, σ_seeds, AUC
│   ├── nexus.py                  # NEXUS — scorer gravitatorio
│   └── framework.py             # clase principal HelixFramework (API pública)
├── tests/
│   ├── test_helix_model.py
│   ├── test_validator.py
│   ├── test_nexus.py
│   └── test_framework.py
├── examples/
│   ├── quick_start.py            # 20 líneas, dataset sintético
│   ├── elliptic_demo.py          # Bitcoin fraud
│   └── amlsim_demo.py            # lavado de dinero
├── docs/
│   └── api.md                    # referencia de API
├── pyproject.toml
└── PLAN.md                       # este archivo
```

---

## API pública (objetivo)

```python
from helix import HelixFramework

# 1. Inicializar y entrenar
fw = HelixFramework()
fw.fit(X, edge_index, y)
# → Validator elige el mejor modelo para este grafo
# → entrena HELIX siempre (para explain/nexus)
# → entrena modelo ganador si no es HELIX

# 2. Inferencia rápida (Laser Query)
scores = fw.predict(X_new, edge_index)
# → retorna array de scores [0,1] por nodo
# → usa laser query: 5× más rápido

# 3. Inferencia con explicación
result = fw.explain(X_new, edge_index)
# → result.scores        — probabilidades
# → result.rho           — inestabilidad local por nodo
# → result.tau           — torque (cuánto "empuja" el campo)
# → result.geo_flag      — True si nodo no rota en S³
# → result.model_used    — qué modelo ganó en este dominio
# → result.confidence    — confianza del Validator

# 4. Scoring gravitatorio (semi-supervisado)
nexus_scores = fw.nexus(X, edge_index, confirmed=[12, 45, 891])
# → propaga desde nodos confirmados como fraude
# → no requiere re-entrenamiento
# → útil con pocos labels conocidos

# 5. Diagnóstico de dominio
report = fw.validate(X, edge_index, y)
# → report.recommended_model
# → report.sigma_seeds
# → report.graph_density
# → report.gini
# → report.reason
```

---

## Tareas — Fase 1

### Semana 1: Núcleo

| # | Tarea | Archivo destino | Fuente |
|---|-------|-----------------|--------|
| 1 | Portar operaciones cuaterniónicas | `helix/core/rotors.py` | `gfcn/rotors.py` |
| 2 | Portar Chebyshev FNO | `helix/core/chebyshev.py` | `gfcn/chebyshev.py` |
| 3 | Portar construcción de Laplaciana sparse | `helix/core/laplacian.py` | `gfcn/chebyshev.py` (normalized_laplacian) |
| 4 | Utilidades de grafo | `helix/core/graph.py` | `stage_graph_validator/run.py` |
| 5 | Modelo HELIX limpio | `helix/models/helix.py` | `stage_diag66_torque_aux/run.py` |
| 6 | Modelo SAGE | `helix/models/sage.py` | `stage_diag69_nids_cyber/run.py` |
| 7 | Modelo GCN | `helix/models/gcn.py` | `stage_diag69_nids_cyber/run.py` |
| 8 | Modelo MLP | `helix/models/mlp.py` | `stage_diag69_nids_cyber/run.py` |

### Semana 2: Framework

| # | Tarea | Archivo destino | Fuente |
|---|-------|-----------------|--------|
| 9  | Graph Validator (regla de 4 condiciones) | `helix/validator.py` | `stage_graph_validator/run.py` |
| 10 | Métricas nativas (ρ, τ, F1_geo, σ_seeds) | `helix/metrics.py` | `stage_diag66_torque_aux/run.py` |
| 11 | NEXUS gravitatorio | `helix/nexus.py` | `stage_diag63_taxonomy_grav/run.py` |
| 12 | Loop de entrenamiento unificado | `helix/trainer.py` | Nuevo |
| 13 | Laser Query | `helix/models/helix.py` (método) | `stage_diag60_laser_query/run.py` |
| 14 | Clase HelixFramework | `helix/framework.py` | Nuevo |

### Semana 3: Tests y ejemplos

| # | Tarea | Archivo destino |
|---|-------|-----------------|
| 15 | Tests modelo HELIX | `tests/test_helix_model.py` |
| 16 | Tests Validator | `tests/test_validator.py` |
| 17 | Tests NEXUS | `tests/test_nexus.py` |
| 18 | Tests framework end-to-end | `tests/test_framework.py` |
| 19 | Ejemplo quick start (sintético) | `examples/quick_start.py` |
| 20 | Ejemplo Elliptic | `examples/elliptic_demo.py` |
| 21 | pyproject.toml | raíz |

---

## Reglas de diseño

1. **Sin dependencias de investigación** — el paquete no importa nada de `GFCN -Graph Fourier Clifford Network/`
2. **Dependencias mínimas** — torch, numpy, scikit-learn, scipy. Sin PyG ni DGL.
3. **explain() solo si dominio compatible** — si σ_seeds ≥ 0.025 o E/N > 10, explain() lanza advertencia
4. **Tipos correctos** — todos los inputs/outputs documentados con tipos Python
5. **Sin prints internos** — usar logging estándar, nivel configurable

---

## Decisiones de diseño pendientes

- [ ] ¿HELIX usa HIDDEN=16 (como Diag#66, mejor AUC) o HIDDEN=3 (más rápido, más interpretable)?
      → Propuesta: HIDDEN=16 por defecto, parámetro configurable
- [x] ¿Triplet Loss se incluye desde el inicio o post-Diag#70?
      → **NO incluir.** Diag#70 confirmó que λ>0 baja AUC (-1.2pp) sin mejorar F1_geo. Baseline sin regularización es mejor.
- [ ] ¿Graph Validator corre automáticamente en fit() o el usuario puede forzar un modelo?
      → Ambas: `fw.fit(X, ei, y, model='auto')` o `fw.fit(X, ei, y, model='helix')`
- [ ] ¿nexus() requiere que HELIX esté entrenado o funciona con cualquier modelo?
      → Requiere HELIX (necesita el espacio S³). Si el Validator eligió SAGE, HELIX también se entrena en paralelo (más lento pero necesario para nexus/explain).

---

## Benchmarks a incluir en README

| Dataset | HELIX | SAGE | GCN | MLP | Modelo recomendado |
|---------|-------|------|-----|-----|--------------------|
| Elliptic | **0.9624** | 0.8833 | 0.8572 | 0.8902 | HELIX |
| AMLSim | **0.9647** | 0.8547 | 0.7927 | 0.9652* | HELIX |
| PaySim | 0.9093 | **0.9768** | 0.9630 | — | SAGE |
| NIDS | 0.8120 | **0.9560** | 0.9523 | 0.7140 | SAGE |

*AMLSim MLP con grafo omitido (split por nodo)
