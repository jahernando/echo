"""Tests for echo.core.Echo — the full train/test pipeline."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from echo import Echo
from echo.synthetic import make_sample


# ---------------------------------------------------------------------------
# train(): whitening on the train sample
# ---------------------------------------------------------------------------


def test_train_outputs_have_expected_shape():
    train = make_sample(2_000, ["normal", "uniform", "normal"], seed=0)

    z, p = Echo().train(train)

    assert z.shape  == (2_000, 3)
    assert p.shape  == (2_000,)
    assert list(z.columns) == ["z0", "z1", "z2"]


def test_train_preserves_index():
    train       = make_sample(500, ["normal", "uniform"], seed=0)
    train.index = pd.RangeIndex(1000, 1500)

    z, p = Echo().train(train)

    assert list(z.index) == list(train.index)
    assert list(p.index) == list(train.index)


def test_train_whitens_the_sample():
    """After train, the transformed sample has mean≈0, var≈1 and cov≈I."""

    train = make_sample(
        20_000,
        ["normal", "uniform", "normal"],
        correlation=0.5,
        seed=1,
    )
    z, _ = Echo().train(train)

    np.testing.assert_allclose(z.mean().to_numpy(), 0.0,             atol=0.05)
    np.testing.assert_allclose(z.var().to_numpy(),  1.0,             atol=0.05)
    np.testing.assert_allclose(z.corr().to_numpy(), np.eye(3),       atol=0.03)


def test_train_pvalues_are_uniform_by_construction():
    """The train ECDF of chi2 makes train p-values uniform on (0, 1)."""

    train  = make_sample(10_000, ["normal", "uniform", "normal"], seed=2)
    _, p   = Echo().train(train)

    stat, _ = kstest(p, "uniform")
    assert stat < 0.02


# ---------------------------------------------------------------------------
# test(): same distribution → uniform p-values
# ---------------------------------------------------------------------------


def test_test_on_h0_sample_is_uniform():
    """An independent sample from the same distribution gives uniform p."""

    train = make_sample(20_000, ["normal", "uniform", "normal"], correlation=0.4, seed=3)
    test  = make_sample(10_000, ["normal", "uniform", "normal"], correlation=0.4, seed=4)

    echo  = Echo()
    echo.train(train)
    _, p_test = echo.test(test)

    assert abs(p_test.mean() - 0.5)            < 0.02
    assert abs((p_test < 0.05).mean() - 0.05)  < 0.015


def test_test_detects_mean_shift():
    """A shifted test sample gives an excess in the low-p tail."""

    train = make_sample(20_000, ["normal", "uniform", "normal"], correlation=0.4, seed=5)
    test  = make_sample(10_000, ["normal", "uniform", "normal"], correlation=0.4, seed=6)

    echo  = Echo()
    echo.train(train)

    shifted      = test.copy()
    shifted["x"] = shifted["x"] + 1.0
    _, p_shift   = echo.test(shifted)

    # Excess of anomalous events in the low-p tail (physics convention).
    assert (p_shift < 0.05).mean() > 0.10


def test_test_preserves_index_and_column_names():
    train      = make_sample(500, ["normal", "uniform"], seed=7)
    test       = make_sample(300, ["normal", "uniform"], seed=8)
    test.index = pd.Index([f"evt_{i}" for i in range(len(test))])

    echo = Echo()
    echo.train(train)
    z, p = echo.test(test)

    assert list(z.columns) == ["z0", "z1"]
    assert list(z.index)   == list(test.index)
    assert list(p.index)   == list(test.index)


# ---------------------------------------------------------------------------
# fitted attributes
# ---------------------------------------------------------------------------


def test_fitted_attributes_have_correct_shapes():
    train = make_sample(1_000, ["normal", "uniform", "normal"], seed=9)
    echo  = Echo()
    echo.train(train)

    assert echo._rotation.shape    == (3, 3)
    assert echo._eigenvalues.shape == (3,)
    assert echo._scales.shape      == (3,)

    # Eigenvalues are sorted descending and positive.
    assert np.all(echo._eigenvalues > 0)
    assert np.all(np.diff(echo._eigenvalues) <= 0)

    # Scales are sqrt of eigenvalues.
    np.testing.assert_allclose(echo._scales, np.sqrt(echo._eigenvalues))


def test_rotation_is_orthonormal():
    train = make_sample(2_000, ["normal", "uniform", "normal"], correlation=0.6, seed=10)
    echo  = Echo()
    echo.train(train)

    np.testing.assert_allclose(echo._rotation.T @ echo._rotation, np.eye(3), atol=1e-10)


def test_uniformizers_one_per_column():
    train = make_sample(500, ["normal", "uniform", "normal"], seed=11)
    echo  = Echo()
    echo.train(train)

    assert set(echo._uniformizers) == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_test_before_train_raises():
    test = make_sample(100, ["normal", "uniform"], seed=12)
    with pytest.raises(RuntimeError, match="not been trained"):
        Echo().test(test)


def test_test_with_missing_columns_raises():
    train = make_sample(500, ["normal", "uniform", "normal"], seed=13)
    test  = make_sample(200, ["normal", "uniform", "normal"], seed=14)
    test  = test.drop(columns=["y"])

    echo = Echo()
    echo.train(train)
    with pytest.raises(ValueError, match="missing columns"):
        echo.test(test)


def test_test_reorders_columns_to_match_train():
    """Columns out of order in test must be reordered before transforming."""

    train          = make_sample(500, ["normal", "uniform"], seed=15)
    test           = make_sample(200, ["normal", "uniform"], seed=16)
    test_reordered = test[["y", "x"]]

    echo = Echo()
    echo.train(train)

    z_ordered,    _ = echo.test(test)
    z_reordered,  _ = echo.test(test_reordered)

    pd.testing.assert_frame_equal(z_ordered, z_reordered)


# ---------------------------------------------------------------------------
# behaviour with d=1
# ---------------------------------------------------------------------------


def test_single_variable_pipeline():
    """The pipeline must work for d=1 (degenerate rotation)."""

    train = make_sample(5_000, ["normal"], seed=17)
    test  = make_sample(2_000, ["normal"], seed=18)

    echo = Echo()
    z_tr, p_tr = echo.train(train)
    z_te, p_te = echo.test(test)

    assert z_tr.shape == (5_000, 1)
    assert z_te.shape == (2_000, 1)

    assert abs(p_te.mean() - 0.5) < 0.03
