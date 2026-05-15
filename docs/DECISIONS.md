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
