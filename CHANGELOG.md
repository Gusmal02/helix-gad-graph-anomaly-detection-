# Changelog — Helix

Registro de cambios significativos por sesión de desarrollo.

---

## [0.2.1] — 2026-08-07

### Cambios

**`_train_result` en `HelixFramework`**
- `fw._train_result` expone el `TrainResult` de HELIX (AUC, epochs_run, loss_history) tras `fit()`.

### Experimentos (Diag#73)

Ablación de regularizadores en Elliptic Bitcoin (203k nodos, 468k aristas).

| Hipótesis | Resultado |
|-----------|-----------|
| `omega_net` per-nodo | **Descartado.** AUC cayó de 0.9624 → 0.8453 (-12pp). Desestabiliza el rotor loop en grafos grandes. |
| `spectral_norm=False` | 0.8897 vs 0.8453 — mejor sin SN cuando la geometría está perturbada, pero ambos por debajo del baseline. |
| `torque_lambda` sweep 0.01→0.20 | Mejora monotónica (0.8596→0.8707) pero no supera el threshold +0.3pp. Ningún λ accionable. |
| `centroid_lambda` sweep 0.05→0.50 | Neutral en valores bajos; destructivo en λ=0.50 (AUC=0.8121). |

**Conclusión:** ningún regularizador mejora el baseline 0.9624 en Elliptic. Los defaults actuales son óptimos para este dominio. `omega_net` revertido del modelo (Diag#47-análogo: hipótesis refutada por datos).

---

## [0.2.0] — 2026-08-03

### Nuevas funcionalidades

**`fw.sonar()` — scorer S³ + distancia de saltos**
- Combina puntuación geodésica NEXUS con distancia BFS desde semillas confirmadas.
- Parámetros: `alpha`, `max_hops`, `hop_decay`. Acepta `alpha='auto'`.
- Disponible directamente en `HelixFramework.sonar()`.

**`fw.save()` / `HelixFramework.load()` — persistencia**
- Guarda pesos, hiperparámetros, componentes PCA y reporte de validación en un único `.pt`.
- Reconstruye el framework completo con `HelixFramework.load(path)`.

**`auto_pca` en `.fit()` — reducción automática de dimensionalidad**
- `fw.fit(X, ei, labels, auto_pca=20)` aplica SVD y reduce features a D componentes antes del entrenamiento.
- Componentes guardados en save/load; transformación aplicada automáticamente en predict/explain/nexus/sonar.
- Motivación: Diag#33 mostró que D=20 vs D=3 da +10pp AUC en Elliptic.

**`directed=True` en `HelixFramework` — Laplaciano dirigido**
- Usa `D_out⁻¹ A` (random-walk) en lugar del Laplaciano simétrico.
- Mejor para grafos transaccionales donde la dirección sender→receiver importa.
- Implementado en `core/laplacian.py` con `directed` flag.

**Early stopping en `TrainConfig`**
- `patience > 0` detiene el entrenamiento cuando la val-loss no mejora en N épocas.
- Restaura el mejor estado al finalizar.
- `min_delta` controla el umbral mínimo de mejora.
- `epochs_run` en `TrainResult` refleja las épocas reales ejecutadas.

**`torque_lambda` en `TrainConfig` — regularización de torque**
- Añade `λ · mean(‖τ‖)` a la pérdida durante entrenamiento.
- Penaliza desplazamientos geométricos excesivos en el campo de rotores.
- `model._last_tau` expone el tensor de torque para acceso externo.

**`result.torque` en `ExplainResult`**
- Campo nuevo `torque: ndarray (N,)` — magnitud del torque por nodo tras el forward completo.
- Señal geométrica interpretable: nodos con torque alto experimentan desplazamiento espectral fuerte.

**Spectral Normalization en `ChebyshevFNO`**
- Cada proyección de filtro envuelta con `torch.nn.utils.spectral_norm`.
- Limita la constante de Lipschitz de cada filtro Chebyshev a ≤ 1 por construcción.
- Previene explosión de autovalores y colapso de frecuencia en dominios heterofílicos.
- Activado por defecto (`spectral_norm=True`). Desactivable para ablaciones.
- Expuesto en `HelixModel(spectral_norm=...)` y `HelixFramework(spectral_norm=...)`.

**`nexus()` y `.nexus()` aceptan `alpha='auto'`**
- Calibra automáticamente α = 1 / mediana(dist_geo(semillas, todos)).
- Recomendado cuando se desconoce la dispersión típica del grafo en S³.

**Centroid Repulsion en S³ — `centroid_lambda` en `TrainConfig`**
- Penaliza `|⟨c_fraud, c_normal⟩|²` — producto interno al cuadrado entre el cuaternión medio de nodos fraude y nodos normales del batch.
- Empuja los centroides de clase hacia ortogonalidad en S³ (90° de separación = pérdida = 0).
- Costo computacional: dos promedios + un producto escalar por batch — prácticamente gratis.
- Solo activo para HELIX (requiere `q_final`); ignorado silenciosamente para SAGE/GCN/MLP.
- Inspirado en la regularización de ortogonalización de centroides (C^H C = I) adaptada a K=2 clases en S³.

### Tests
- Suite ampliada de 57 → 69 tests.
- Nuevos grupos: `sonar` framework, `save/load` (HELIX y SAGE), `auto_pca`, early stopping, `directed`, `torque`, `torque_lambda`, `spectral_norm`.

### Documentación
- `docs/api.md`: secciones NEXUS, SONAR, `sparsify_top_k` completas.
- `README.md`: Quick Start, API, TrainConfig, Directed Graphs, Spectral Normalization actualizados.

---

## [0.1.1] — 2026-08-02

### Correcciones

**`laser_query` — firma corregida**
- Bug: `helix_report.qmd` y `helix_guide.ipynb` pasaban el Laplaciano pre-computado `L` en lugar del `edge_index`.
- Fix: `model.laser_query(X_t, ei_t, ew_t)` — firma correcta: `(x, edge_index, edge_weight)`.

**`helix_guide.ipynb` — reescritura completa**
- Archivo original contenía dos notebooks concatenados (JSON inválido).
- Reescrito desde cero con 10 celdas: setup, elliptic, train, geometric viz, laser query, NEXUS+SONAR, save/load, resultados Diag#71.

### Tests
- 19 tests en `test_nexus.py` cubriendo `sonar_score`, `sonar_score_normalized`, `_auto_alpha`, `_bfs_min_hops`.

---

## [0.1.0] — 2026-07-xx

### Lanzamiento inicial

**Arquitectura base**
- `EmissionMLP(D→16→3)` → `ChebyshevFNO(K=4, residual_mix=True)` → Rotor loop T=5 en S³ → `Linear(3,1)`.
- Cuaterniones normalizados a ‖q‖=1 en cada paso (S³).
- `laser_query`: inferencia rápida con 1 paso FNO (≈5× más rápido, AUC equivalente).

**Graph Validator — 4 reglas**
- Regla 1: E/N > 10 → SAGE
- Regla 2: Gini > 0.6 AND homofilia > 0.95 → SAGE
- Regla 3: σ_seeds ≥ 0.025 AND lift < −0.05 → MLP; σ ≥ 0.025 → SAGE
- Regla 4: σ < 0.025 AND E/N ≤ 10 → HELIX
- Precisión empírica: 4/4 dominios (Elliptic, AMLSim, PaySim, Electricity).

**NEXUS — scorer gravitacional semi-supervisado**
- `nexus(j) = Σ exp(−α · arccos(|⟨qᵢ, qⱼ⟩|))`
- Sin reentrenamiento. Propaga riesgo desde nodos confirmados por S³.

**Benchmarks validados**
| Dataset | Helix | SAGE | GCN | MLP |
|---------|-------|------|-----|-----|
| Elliptic (Bitcoin) | **0.9624** | 0.8833 | 0.8572 | 0.8902 |
| AMLSim (lavado dinero) | **0.9647** | 0.8547 | 0.7927 | 0.9652 |
| PaySim (fraude móvil) | 0.9093 | **0.9768** | 0.9630 | — |
| NF-UQ-NIDS (ciberseg) | 0.8120 | **0.9560** | 0.9523 | 0.7140 |

**Ley empírica validada**
- Grafos causales dispersos (E/N < 5) → Helix gana.
- Grafos densos o kNN artificiales → SAGE/MLP ganan.
- Pearson(n_raw_features, graph_lift) = −0.587.
