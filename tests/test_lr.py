"""Tests for echo.lr.score_lr."""

import numpy as np
import pandas as pd
import pytest

from echo import Echo, score_lr
from echo.synthetic import make_sample


def _fit_two_echos():
    """Build two fitted Echo instances on clearly different H0 and H1 samples."""

    h0 = make_sample(
        20_000, ["normal", "uniform", "normal"],
        parameters=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        correlation=0.5, seed=1,
    )
    h1 = make_sample(
        20_000, ["normal", "uniform", "normal"],
        parameters=[(1.5, 2.0), (-2.0, 3.0), (-0.5, 0.5)],
        correlation=-0.2, seed=2,
    )
    echo_h0 = Echo(); echo_h0.train(h0)
    echo_h1 = Echo(); echo_h1.train(h1)
    return echo_h0, echo_h1


# ---------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------


def test_score_lr_returns_series_with_sample_index():
    echo_h0, echo_h1 = _fit_two_echos()
    sample          = make_sample(500, ["normal", "uniform", "normal"], seed=10)
    sample.index    = pd.Index([f"evt_{i}" for i in range(len(sample))])

    score = score_lr(echo_h1, echo_h0, sample)

    assert isinstance(score, pd.Series)
    assert score.name == "delta_chi2"
    assert list(score.index) == list(sample.index)
    assert score.shape == (500,)


def test_score_lr_negative_for_h0_sample():
    """A sample drawn from H0 should have Δχ² < 0 on average (more H0-like)."""

    echo_h0, echo_h1 = _fit_two_echos()
    sample_h0 = make_sample(
        5_000, ["normal", "uniform", "normal"],
        parameters=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        correlation=0.5, seed=20,
    )

    score = score_lr(echo_h1, echo_h0, sample_h0)
    assert score.median() < 0


def test_score_lr_positive_for_h1_sample():
    """A sample drawn from H1 should have Δχ² > 0 on average (more H1-like)."""

    echo_h0, echo_h1 = _fit_two_echos()
    sample_h1 = make_sample(
        5_000, ["normal", "uniform", "normal"],
        parameters=[(1.5, 2.0), (-2.0, 3.0), (-0.5, 0.5)],
        correlation=-0.2, seed=21,
    )

    score = score_lr(echo_h1, echo_h0, sample_h1)
    assert score.median() > 0


def test_score_lr_discriminates_well():
    """ROC AUC of Δχ² distinguishing H0 vs H1 samples should be high."""

    from sklearn.metrics import roc_auc_score

    echo_h0, echo_h1 = _fit_two_echos()
    sample_h0 = make_sample(
        5_000, ["normal", "uniform", "normal"],
        parameters=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        correlation=0.5, seed=30,
    )
    sample_h1 = make_sample(
        5_000, ["normal", "uniform", "normal"],
        parameters=[(1.5, 2.0), (-2.0, 3.0), (-0.5, 0.5)],
        correlation=-0.2, seed=31,
    )

    d0  = score_lr(echo_h1, echo_h0, sample_h0).to_numpy()
    d1  = score_lr(echo_h1, echo_h0, sample_h1).to_numpy()
    auc = roc_auc_score(
        np.concatenate([np.zeros_like(d0), np.ones_like(d1)]),
        np.concatenate([d0, d1]),
    )
    assert auc > 0.9


def test_score_lr_sign_flips_when_echos_swap():
    """Swapping the two echos must flip the sign of Δχ² exactly."""

    echo_h0, echo_h1 = _fit_two_echos()
    sample          = make_sample(500, ["normal", "uniform", "normal"], seed=40)

    forward  = score_lr(echo_h1, echo_h0, sample)
    backward = score_lr(echo_h0, echo_h1, sample)
    np.testing.assert_allclose(forward.to_numpy(), -backward.to_numpy())


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_score_lr_requires_trained_echos():
    untrained = Echo()
    trained   = Echo()
    trained.train(make_sample(500, ["normal", "uniform"], seed=50))
    sample    = make_sample(100, ["normal", "uniform"], seed=51)

    with pytest.raises(RuntimeError, match="not been trained"):
        score_lr(untrained, trained, sample)
    with pytest.raises(RuntimeError, match="not been trained"):
        score_lr(trained, untrained, sample)
