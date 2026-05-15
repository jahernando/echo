"""First two steps of the *echo* transform pipeline.

Step 1 — uniformize: map a 1D variable to (0, 1) via the empirical CDF of a
training sample.

Step 2 — to_normal: map a uniform variable to a standard normal via the probit
(inverse normal CDF).
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import norm


def fit_uniformize(train) -> Callable[[np.ndarray], np.ndarray]:
    """Fit an empirical-CDF uniformizer on a 1D training sample.

    Parameters
    ----------
    train : array_like, shape (n_train,)
        Reference sample defining the ECDF.

    Returns
    -------
    transform : callable
        Function ``u = transform(x)`` mapping any array ``x`` to its quantile
        in the training ECDF. Output lies strictly in (0, 1) thanks to a
        midrank-with-pseudocount convention, so the result can be safely fed
        into ``to_normal`` without producing infinities at the tails.

    Notes
    -----
    The convention is ``u = (rank + 0.5) / (n_train + 1)`` where ``rank`` is
    the count of training points strictly less-or-equal to ``x``. This gives
    ``u ∈ (0.5/(n+1), (n+0.5)/(n+1)) ⊂ (0, 1)`` for any input.
    """

    sorted_train = np.sort(np.asarray(train).ravel())
    n_train      = sorted_train.size

    if n_train == 0:
        raise ValueError("train sample is empty")

    def transform(x):
        x_arr = np.asarray(x)
        ranks = np.searchsorted(sorted_train, x_arr, side="right")
        return (ranks + 0.5) / (n_train + 1)

    return transform


def to_normal(u) -> np.ndarray:
    """Map a uniform variable to a standard normal via the probit transform.

    Parameters
    ----------
    u : array_like
        Values in (0, 1). Inputs are clipped to ``(eps, 1 - eps)`` with
        ``eps = 1e-12`` to avoid ``±inf`` at the boundaries.

    Returns
    -------
    ndarray
        ``norm.ppf(u)``, i.e. the inverse standard-normal CDF applied
        elementwise.
    """

    u_arr     = np.asarray(u, dtype=float)
    u_clipped = np.clip(u_arr, 1e-12, 1.0 - 1e-12)
    return norm.ppf(u_clipped)
