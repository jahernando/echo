# Decisions log — echo

Registro de decisiones de diseño / convenciones adoptadas en el proyecto. Cada
entrada incluye el contexto, la decisión y el razonamiento. Ordenado
cronológicamente, las decisiones nuevas se añaden al final.

El criterio: registrar aquí sólo lo que no se deduce del código y que un
colaborador (o yo mismo dentro de 6 meses) querría poder justificar.

---

## D1 — Arquitectura: funciones puras + clase orquestadora

**Decisión.** Los pasos del algoritmo se implementan como funciones puras en
`echo.transform` (y módulos análogos para PCA, whitening y chi2). La clase
`Echo` (`echo.core`) sólo encadena esas funciones y guarda los atributos
ajustados durante `train`.

**Razón.** Las funciones quedan testeables aisladas y reutilizables fuera del
pipeline (notebooks exploratorios, comparativas con otros enfoques). La clase
añade una fachada con estado sin esconder la API funcional.

## D2 — Convención de la uniformización (paso 1)

**Decisión.** La ECDF empírica del train se evalúa como
`u = (rank + 0.5) / (n_train + 1)`, donde `rank` es el número de puntos del
train con valor ≤ al evaluado. El output queda estrictamente en
`(0.5/(n+1), (n+0.5)/(n+1)) ⊂ (0, 1)`.

**Razón.** Garantiza `u ∈ (0, 1)` también para valores fuera del rango del
train, evitando que `to_normal` (probit) reciba 0 o 1 y devuelva `±inf`. La
pseudocuenta `+0.5` es la convención de mid-rank estándar.

## D3 — Probit con clipping defensivo (paso 2)

**Decisión.** `to_normal` aplica `np.clip(u, 1e-12, 1 − 1e-12)` antes del
`norm.ppf`.

**Razón.** Aunque `fit_uniformize` ya entrega `u` en `(0, 1)` por D2, los
usuarios pueden llamar a `to_normal` directamente con `u = 0` o `u = 1` (p.
ej. desde otros uniformizadores). El clipping evita `±inf` sin asumir el
origen del input.

## D4 — Datos sintéticos con cópula gaussiana

**Decisión.** `make_sample` genera variables correladas vía cópula gaussiana:
muestrea normales multivariantes con la matriz de correlación pedida y luego
mapea cada columna a su marginal por inverso de CDF.

**Razón.** Permite combinar marginales arbitrarias con una estructura de
dependencia controlada. Implicación: la correlación de Pearson del output
queda atenuada para marginales no-normales, pero la correlación de rango
(Spearman) coincide exactamente con la matriz pedida. Esto se documenta en
la docstring de `make_sample`.

## D5 — Convención de parámetros por distribución

**Decisión.** Las tuplas de `parameters=` en `make_sample` siguen:

- `"normal"`: `(mean, std)`, con `std > 0`.
- `"uniform"`: `(low, high)`, con `high > low`.

**Razón.** Para `normal` coincide con scipy (`loc`, `scale`). Para `uniform`
se prefiere `(low, high)` sobre el `(loc, scale)` de scipy porque es más
legible cuando uno escribe los parámetros a mano (`(-3, 7)` lee mejor que
`(-3, 10)`). Coste: pequeña inconsistencia con scipy, asumida.

## D6 — Nombres de los métodos: `train` / `test`

**Decisión.** La clase `Echo` expone `train(sample)` y `test(sample)` en
lugar del clásico `fit(X)` / `transform(X)` de scikit-learn.

**Razón.** Refleja la semántica del dominio (comparación de dos muestras
*train* vs *test*) y coincide con el lenguaje que el usuario emplea para
describir el algoritmo. Coste: salirse de la convención sklearn; aceptado
porque `Echo` no se integra en pipelines de sklearn.

## D7 — PCA vía `np.linalg.eigh` sobre la covarianza, ordenado descendente

**Decisión.** Pasos 3–5 usan eigendecomposición simétrica de
`np.cov(z, rowvar=False)`. Autovalores y autovectores se ordenan descendente
(la dirección de mayor varianza es `z0`).

**Razón.** Determinista, suficiente para `d` pequeño (≤ decenas de variables)
y reproducible. Si en algún momento `d` crece a centenares, se reevalúa con
SVD truncado.

## D8 — p-value en convención física (paso 8)

**Decisión.** `p = 1 − ECDF_train(chi2)`. Eventos con chi2 alto (lejos del
origen en el espacio whitened) tienen `p` pequeño.

**Razón.** Convención estándar en física de partículas: "p pequeño = evento
anómalo". Bajo H0 la distribución es U(0, 1) igualmente, lo cual mantiene
válidos los tests de uniformidad. El paso 8 de la descripción original del
algoritmo decía literalmente "uniformizar chi2 para usar como p-value"; se
fija aquí la convención de signo.

## D9 — Nombres de columnas tras la transformación

**Decisión.** Los outputs de `Echo.train` / `Echo.test` usan columnas
`z0, z1, …, z{d−1}`.

**Razón.** Tras la rotación PCA las componentes ya no corresponden a las
variables originales, así que reutilizar los nombres `x, y, z` sería
engañoso. El prefijo `z` indica "espacio whitened (normal estándar)".

## D10 — Interfaz: pandas DataFrame

**Decisión.** `Echo.train` y `Echo.test` aceptan `DataFrame` o array_like, y
siempre devuelven `(DataFrame, Series)`. El índice del input se preserva en
ambos outputs.

**Razón.** Mantener el índice permite alinear los p-values con metadatos
externos (etiquetas, pesos, IDs de evento) sin gestión adicional.

## D11 — Diagnóstico del paso 6: `Echo.diagnose(z)`

**Decisión.** El paso 6 (verificación de que las variables transformadas son
N(0, I) conjuntas) se implementa como `Echo.diagnose(z, deep=False)`, fuera
del pipeline de transformación. Devuelve un dict con:

- `"marginals"` — DataFrame por componente con `mean`, `std`, `skew`,
  `excess_kurtosis`, `ks_stat` y `ks_pvalue` (KS contra N(0, 1)).
- `"spearman"` — matriz de correlación de Spearman.
- `"iterated_eigenvalues"` (sólo con `deep=True`) — autovalores tras re-aplicar
  uniformize+probit+PCA sobre `z`.

**Razón.**

1. **Marginales**: el control mínimo y barato. KS, skew y kurtosis cubren
   desviaciones de N(0, 1) en cada eje.

2. **Spearman, no Pearson**: tras PCA + whitening sobre el train, la matriz de
   Pearson es *exactamente* la identidad por construcción (los autovectores
   diagonalizan la covarianza) — por tanto no informa. Spearman captura
   dependencia monótona no-lineal que PCA no toca.

3. **Iterated eigenvalues**: re-aplicar el pipeline echo sobre `z` re-marginaliza
   cada componente a N(0, 1) *exacta* y re-diagonaliza. Si `z` ya era
   conjuntamente gaussiana, esto es un near-no-op y los autovalores quedan
   ≈ 1. Si hay dependencia no lineal residual, los autovalores se separan de
   1. Funciona como test de gausianidad conjunta. Marcado como `deep=True`
   por su coste extra y porque para muchos usos las marginales + Spearman ya
   bastan.

`diagnose` toma `z` como argumento explícito (en vez de operar sobre un
atributo `_z_train` guardado): el usuario ya tiene el DataFrame de
`train(...)` o `test(...)`, y así la misma función sirve para auditar el
train y el test sin duplicar API.

## D12 — `compare(test_sample)` y cacheo de `_z_train`

**Decisión.** Se añade `Echo.compare(test_sample, alphas=(0.01, 0.05, 0.10))`
que combina la transformación del test con un set de tests 1D train-vs-test
en el espacio whitened. Devuelve un dict con:

- `"z"`, `"p"` — los outputs estándar de `test(...)`, para reuso.
- `"marginals"` — DataFrame por componente `z_i` con KS de dos muestras
  (train vs test) y su p-valor.
- `"global"` — `mean_p`, `ks_stat_uniform`, `ks_pvalue_uniform` (KS de
  `p_test` vs U(0,1)) y `frac_below_alpha` (Series indexada por `alphas`).

Para evitar duplicar cómputo, `train(...)` cachea ahora el DataFrame `z_train`
en el atributo privado `_z_train` (≈ O(n_train · d) extra de memoria).

**No** se cachea `_p_train`: matemáticamente es determinístico
(`(rank + 0.5) / (n_train + 1)` de los `chi2_train`), y no se usa en
`compare` (la uniformidad del test se valida contra `U(0,1)` teórica, no
contra `p_train`).

**Razón.** `test` y `compare` se diferencian en intención: `test` es la
transformación pura, `compare` es la inferencia (test estadístico) sobre el
test. Separarlos mantiene `test` ligero y deja `compare` como el punto único
para preguntas de tipo "¿se parece este test al train?". El cacheo de
`z_train` es la mínima información que permite responder esa pregunta sin
re-transformar el train cada vez.

## D13 — Discriminación H1 vs H0 con dos `Echo`: función libre `score_lr`

**Decisión.** La discriminación entre dos hipótesis (entrenar un `Echo` sobre
una muestra H1 y otro sobre H0, y puntuar eventos según se parecen más a una
o a otra) se implementa como **función libre** `score_lr(echo_h1, echo_h0,
sample)` en el módulo `echo.lr`, no como método de `Echo`.

Convención de signo:

```
Δχ²(x) = χ²_H0(x) − χ²_H1(x)
       > 0  →  x más H1-like
       < 0  →  x más H0-like
```

Bajo gaussianidad aproximada (el régimen que `echo` busca conseguir tras los
pasos 1–5), `Δχ²` es `2·log(L_H1/L_H0)` salvo constante aditiva, lo cual lo
convierte en un proxy del log-likelihood-ratio.

**Razón para función libre y no método.**

1. La operación es **simétrica** en sus dos `Echo`: ninguno tiene
   preferencia sobre el otro. Convertir uno en `self` introduce una
   asimetría artificial que invita a errores de signo
   (`echo_H0.score_against(echo_H1, ...)` daría el signo opuesto).

2. La firma `score_lr(echo_h1=, echo_h0=)` con kwargs hace explícito quién
   es quién y elimina cualquier ambigüedad de orden.

3. Preferencia general del proyecto: cuando una operación no necesita
   estado interno del objeto, exponerla como función libre. Aplicación de
   la misma filosofía que [[D1]] (funciones puras + clase orquestadora).

**Pendiente.** La versión "proper" del LR (con jacobianos de PIT+probit y
factor `|Σ|`) que daría la ratio de densidades en `ℝ^d` no está implementada
— se deja para una iteración futura si la versión `Δχ²` resulta
insuficiente.
