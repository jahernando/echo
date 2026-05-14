# Style guide — echo

Convenciones de estilo para este repositorio. Complementa las reglas que ya impone `ruff check` (configuradas en `pyproject.toml`).

> **Lo que comprueba ruff automáticamente**: espacios alrededor de operadores, imports no usados, variables no definidas, orden de imports (isort), modernización de sintaxis (`pyupgrade`), bugs comunes (`bugbear`).
>
> **Lo que esta guía cubre**: preferencias estéticas que ruff no comprueba — alineamiento, líneas en blanco, organización visual.

---

## Prioridad

**Legibilidad humana > minimalismo de diff.** Cuando haya conflicto, gana el humano que va a leer el código dentro de seis meses.

## Alineamiento

Alinear `=` en bloques de asignaciones **relacionadas** cuando mejore la lectura comparada:

```python
# Sí:
x_min   = data.x.min()
x_max   = data.x.max()
x_mean  = data.x.mean()

# No (asignaciones no relacionadas):
result      = compute(data)
flag        = True
n_samples   = 100
```

Lo mismo en diccionarios con claves cortas y valores cortos:

```python
config = {
    "n_bins":    50,
    "alpha":     0.05,
    "method":    "ks",
}
```

No forzar alineamiento si la separación crece tanto que rompe el flujo de lectura (más de ~4 espacios entre símbolo y `=`).

## Líneas en blanco

- **2 líneas** antes de funciones/clases top-level (PEP 8 estándar).
- **1 línea** entre métodos de una clase.
- **1 línea en blanco después de `def …:` o tras la docstring** cuando la función sea no trivial:

```python
def compare_marginals(sample_a, sample_b, n_bins=50):
    """Compare 1D marginal distributions via Kolmogorov–Smirnov."""

    assert sample_a.ndim == 1
    assert sample_b.ndim == 1

    statistic, pvalue = ks_2samp(sample_a, sample_b)
    return statistic, pvalue
```

- Usar líneas en blanco internas para separar **fases lógicas** dentro de una función larga (preparación / cómputo / resultado).

## Espacios alrededor de comparadores y operadores

Siempre espacio antes y después: `x == y`, `i < n`, `a + b`. Ruff lo enforza vía `E225`, no hace falta vigilar.

Excepción: argumentos por defecto sin anotación de tipo (`def foo(x=1)`, sin espacios), y exponentes simples (`x**2`). Convención PEP 8 estándar.

## Imports

Tres bloques separados por línea en blanco, en orden:

1. Stdlib (`import os`, `from pathlib import Path`)
2. Third-party (`import numpy as np`)
3. Local (`from echo.metrics import ks_distance`)

Ruff (regla `I`) ordena dentro de cada bloque automáticamente con `ruff check --fix`.

## Nombres

- Funciones, variables, módulos: `snake_case`.
- Clases: `PascalCase`.
- Constantes: `UPPER_SNAKE_CASE`.
- Privados (no exportar): prefijo `_`.
- Evitar abreviaturas crípticas; preferir `n_samples` a `n`, `distribution` a `dist`.

## Docstrings

Una línea para funciones simples. Para funciones más complejas, estilo NumPy:

```python
def ks_distance(a, b):
    """Two-sample Kolmogorov–Smirnov distance.

    Parameters
    ----------
    a, b : array_like
        1D samples to compare.

    Returns
    -------
    float
        Maximum absolute difference between empirical CDFs.
    """
    ...
```

## Comentarios

Sólo cuando expliquen un **porqué** no obvio (decisión, restricción oculta, workaround). No comentar el **qué** — eso lo dice el código.

## Notebooks

Los notebooks (`notebooks/`) son exploratorios. No aplicamos esta guía estrictamente en ellos — ruff los ignora en `E` y `F` por configuración.
