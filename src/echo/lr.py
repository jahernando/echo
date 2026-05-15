"""Likelihood-ratio-style discrimination between two fitted Echo instances.

Given an ``Echo`` fitted on an H1 (signal) sample and another fitted on H0
(background), this module provides ``score_lr`` to score events as
"more H1-like" vs "more H0-like".

The score is ``Δχ²(x) = χ²_H0(x) − χ²_H1(x)``, evaluated event-by-event. Under
the (approximate) Gaussianity that the echo pipeline tries to achieve, this is
``2 · log(L_H1(x) / L_H0(x))`` up to an additive constant — a proxy of the
log-likelihood ratio. Sign convention follows the standard physics use:

    Δχ² > 0  →  event is more H1-like
    Δχ² < 0  →  event is more H0-like

The proper density-based LR (with PIT/probit jacobians and the |Σ| terms) is
out of scope here; this lightweight score is the first practical step.
"""

import pandas as pd


def score_lr(echo_h1, echo_h0, sample):
    """Score events with Δχ² between two fitted Echo instances.

    Parameters
    ----------
    echo_h1 : Echo
        Fitted on the H1 sample.
    echo_h0 : Echo
        Fitted on the H0 sample.
    sample : pandas.DataFrame or array_like
        Events to score. Must have the columns expected by both ``echo_h1``
        and ``echo_h0`` (typically the same observables in both).

    Returns
    -------
    pandas.Series
        Per-event ``Δχ² = χ²_H0(x) − χ²_H1(x)``, indexed like ``sample``.
    """

    z_h1, _ = echo_h1.test(sample)
    z_h0, _ = echo_h0.test(sample)

    chi2_h1 = (z_h1 ** 2).sum(axis=1)
    chi2_h0 = (z_h0 ** 2).sum(axis=1)

    delta = chi2_h0 - chi2_h1
    return pd.Series(delta.to_numpy(), index=z_h1.index, name="delta_chi2")
