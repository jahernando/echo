"""Tests for echo.synthetic.make_sample."""

import numpy as np
import pandas as pd
import pytest

from echo.synthetic import make_sample


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_default_normal_is_standard():
    df = make_sample(20_000, ["normal"], seed=0)

    assert abs(df["x"].mean())       < 0.05
    assert abs(df["x"].std() - 1.0)  < 0.05


def test_default_uniform_is_zero_one():
    df = make_sample(20_000, ["uniform"], seed=0)

    assert df["x"].min() >= 0.0
    assert df["x"].max() <= 1.0
    assert abs(df["x"].mean() - 0.5) < 0.02


def test_shape_and_column_names():
    df = make_sample(100, ["normal", "uniform", "normal"], seed=0)

    assert df.shape == (100, 3)
    assert list(df.columns) == ["x", "y", "z"]


def test_seed_is_reproducible():
    a = make_sample(500, ["normal", "uniform"], seed=42)
    b = make_sample(500, ["normal", "uniform"], seed=42)

    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# user-provided parameters
# ---------------------------------------------------------------------------


def test_normal_parameters_are_respected():
    df = make_sample(
        50_000,
        ["normal"],
        parameters=[(5.0, 2.0)],
        seed=1,
    )

    assert abs(df["x"].mean() - 5.0) < 0.05
    assert abs(df["x"].std()  - 2.0) < 0.05


def test_uniform_parameters_are_respected():
    df = make_sample(
        50_000,
        ["uniform"],
        parameters=[(-3.0, 7.0)],
        seed=1,
    )

    assert df["x"].min() >= -3.0
    assert df["x"].max() <=  7.0
    assert abs(df["x"].mean() - 2.0) < 0.05   # midpoint = (-3 + 7) / 2


def test_mixed_distributions_with_custom_parameters():
    df = make_sample(
        50_000,
        ["normal", "uniform", "normal"],
        parameters=[(10.0, 0.5), (0.0, 100.0), (-1.0, 3.0)],
        seed=2,
    )

    assert abs(df["x"].mean() - 10.0) < 0.05
    assert abs(df["x"].std()  -  0.5) < 0.05

    assert df["y"].min() >= 0.0
    assert df["y"].max() <= 100.0

    assert abs(df["z"].mean() - (-1.0)) < 0.1
    assert abs(df["z"].std()  -  3.0)   < 0.1


def test_parameters_none_uses_defaults():
    """Passing ``parameters=None`` is equivalent to omitting the argument."""

    a = make_sample(500, ["normal", "uniform"], parameters=None, seed=7)
    b = make_sample(500, ["normal", "uniform"], seed=7)

    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------


def test_correlation_recovered_for_normals():
    """Pearson correlation of normal columns matches the latent correlation."""

    df = make_sample(
        50_000,
        ["normal", "normal"],
        parameters=[(1.0, 2.0), (-3.0, 0.5)],
        correlation=0.7,
        seed=4,
    )

    rho = df.corr().iloc[0, 1]
    assert abs(rho - 0.7) < 0.02


def test_correlation_zero_gives_near_independence():
    df = make_sample(20_000, ["normal", "uniform"], correlation=0.0, seed=5)

    assert abs(df.corr().iloc[0, 1]) < 0.03


def test_full_correlation_matrix_accepted():
    target = np.array([
        [1.0, 0.3, 0.6],
        [0.3, 1.0, 0.2],
        [0.6, 0.2, 1.0],
    ])
    df = make_sample(50_000, ["normal", "normal", "normal"], correlation=target, seed=6)

    np.testing.assert_allclose(df.corr().to_numpy(), target, atol=0.02)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_empty_distributions_raises():
    with pytest.raises(ValueError, match="empty"):
        make_sample(100, [])


def test_unknown_distribution_raises():
    with pytest.raises(ValueError, match="unknown distribution"):
        make_sample(100, ["lognormal"])


def test_parameters_length_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        make_sample(100, ["normal", "uniform"], parameters=[(0.0, 1.0)])


def test_normal_std_must_be_positive():
    with pytest.raises(ValueError, match="std must be > 0"):
        make_sample(100, ["normal"], parameters=[(0.0, -1.0)])


def test_uniform_requires_high_above_low():
    with pytest.raises(ValueError, match="high > low"):
        make_sample(100, ["uniform"], parameters=[(5.0, 5.0)])


def test_correlation_matrix_wrong_shape_raises():
    bad = np.eye(2)
    with pytest.raises(ValueError, match="shape"):
        make_sample(100, ["normal", "normal", "normal"], correlation=bad)
