"""Synthetic-data generator for testing the echo pipeline.

Generates a ``pandas.DataFrame`` whose columns follow specified marginal
distributions and share a target dependence structure built via a Gaussian
copula.

The construction:

1. Draw correlated standard normals with the target correlation matrix.
2. Map each column to its requested marginal via the appropriate inverse-CDF.

Note that for non-Gaussian marginals the *Pearson* correlation of the output
will be slightly attenuated relative to the input ``correlation`` — the input
is exactly the correlation of the underlying latent normals, which equals the
*rank* (Spearman) correlation of the output.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

_DEFAULT_NAMES = ("x", "y", "z", "w", "v", "u", "t", "s")

# Per supported distribution: default parameter tuple and a short description
# of what the tuple means. Kept here so it stays in sync with the validators
# and the docstring.
_DIST_DEFAULTS = {
    "normal":  (0.0, 1.0),   # (mean, std)
    "uniform": (0.0, 1.0),   # (low, high)
}


def make_sample(
    n_samples,
    distributions,
    parameters=None,
    correlation=0.0,
    seed=None,
):
    """Generate a DataFrame with given marginals and dependence structure.

    Parameters
    ----------
    n_samples : int
        Number of rows.
    distributions : sequence of str
        One entry per column. Supported values: ``"normal"``, ``"uniform"``.
    parameters : sequence of tuple, optional
        One tuple per column, matching ``distributions``. Conventions:

        - ``"normal"``: ``(mean, std)``, ``std > 0``.
        - ``"uniform"``: ``(low, high)``, ``high > low``.

        If ``None``, defaults are used: normal → ``(0, 1)``, uniform → ``(0, 1)``.
    correlation : float or array_like, default 0.0
        If scalar, used as every off-diagonal entry of an otherwise-identity
        correlation matrix; must lie within the positive-definite range for
        the requested ``n_vars``. If array_like, interpreted as a full
        ``(n_vars, n_vars)`` correlation matrix.
    seed : int or None, optional
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    pandas.DataFrame
        Columns named ``"x"``, ``"y"``, ``"z"``, ... in order, one per entry
        of ``distributions``.
    """

    rng    = np.random.default_rng(seed)
    n_vars = len(distributions)

    if n_vars == 0:
        raise ValueError("distributions must not be empty")
    if n_vars > len(_DEFAULT_NAMES):
        raise ValueError(f"only up to {len(_DEFAULT_NAMES)} variables supported")

    parameters = _resolve_parameters(distributions, parameters)
    corr       = _build_correlation_matrix(correlation, n_vars)

    # Correlated standard normals via Cholesky.
    latent = rng.standard_normal(size=(n_samples, n_vars))
    chol   = np.linalg.cholesky(corr)
    latent = latent @ chol.T

    # Apply the Gaussian copula: convert each latent column to its requested
    # marginal by composing norm.cdf with the appropriate inverse CDF.
    columns = {}
    for i, (dist, params) in enumerate(zip(distributions, parameters)):
        name           = _DEFAULT_NAMES[i]
        columns[name]  = _apply_marginal(latent[:, i], dist, params)

    return pd.DataFrame(columns)


def _apply_marginal(latent, dist, params):
    """Map a standard-normal column to the requested marginal via the copula."""

    if dist == "normal":
        mean, std = params
        return mean + std * latent

    if dist == "uniform":
        low, high = params
        u = norm.cdf(latent)
        return low + (high - low) * u

    raise ValueError(f"unknown distribution {dist!r}; supported: {tuple(_DIST_DEFAULTS)}")


def _resolve_parameters(distributions, parameters):
    """Fill in defaults, validate length and per-distribution constraints."""

    if parameters is None:
        parameters = [_DIST_DEFAULTS[d] if d in _DIST_DEFAULTS else None
                      for d in distributions]
    else:
        parameters = list(parameters)

    if len(parameters) != len(distributions):
        raise ValueError(
            f"len(parameters)={len(parameters)} does not match "
            f"len(distributions)={len(distributions)}"
        )

    for dist, params in zip(distributions, parameters):
        if dist not in _DIST_DEFAULTS:
            raise ValueError(f"unknown distribution {dist!r}; supported: {tuple(_DIST_DEFAULTS)}")
        _validate_params(dist, params)

    return parameters


def _validate_params(dist, params):
    """Raise ValueError if ``params`` is malformed for ``dist``."""

    if len(params) != 2:
        raise ValueError(f"{dist!r} expects a 2-tuple, got {params!r}")

    if dist == "normal":
        _, std = params
        if std <= 0:
            raise ValueError(f"normal std must be > 0, got {std}")
    elif dist == "uniform":
        low, high = params
        if high <= low:
            raise ValueError(f"uniform requires high > low, got low={low}, high={high}")


def _build_correlation_matrix(correlation, n_vars):
    """Return an (n_vars, n_vars) correlation matrix from a scalar or array."""

    if np.isscalar(correlation):
        corr = np.full((n_vars, n_vars), float(correlation))
        np.fill_diagonal(corr, 1.0)
        return corr

    corr = np.asarray(correlation, dtype=float)
    if corr.shape != (n_vars, n_vars):
        raise ValueError(f"correlation matrix must have shape ({n_vars}, {n_vars})")
    return corr
