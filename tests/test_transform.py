"""Tests for echo.transform — steps 1 (uniformize) and 2 (to_normal)."""

import numpy as np
import pytest
from scipy.stats import kstest, norm

from echo.transform import fit_uniformize, to_normal


# ---------------------------------------------------------------------------
# fit_uniformize
# ---------------------------------------------------------------------------


def test_uniformize_output_in_open_unit_interval():
    """The uniformized output must lie strictly inside (0, 1)."""

    rng    = np.random.default_rng(0)
    train  = rng.normal(size=1000)
    sample = rng.normal(size=500)

    u = fit_uniformize(train)(sample)

    assert (u > 0).all()
    assert (u < 1).all()


def test_uniformize_extreme_inputs_stay_in_open_interval():
    """Values far below / above the train range still map into (0, 1)."""

    train = np.linspace(-1.0, 1.0, 100)
    transform = fit_uniformize(train)

    out_of_range = np.array([-1e6, -1e3, 1e3, 1e6])
    u = transform(out_of_range)

    assert (u > 0).all()
    assert (u < 1).all()


def test_uniformize_is_monotone_non_decreasing():
    """Sorting the input must not change the order of the output."""

    rng    = np.random.default_rng(1)
    train  = rng.normal(size=2000)
    sample = rng.uniform(-5, 5, size=400)

    transform = fit_uniformize(train)
    u_sorted  = transform(np.sort(sample))

    assert np.all(np.diff(u_sorted) >= 0)


def test_uniformize_train_on_itself_is_uniform():
    """Applying the transform to the train sample yields ≈ U(0, 1)."""

    rng   = np.random.default_rng(2)
    train = rng.normal(size=10_000)

    u = fit_uniformize(train)(train)

    stat, _ = kstest(u, "uniform")
    assert stat < 0.02


def test_uniformize_independent_sample_is_uniform():
    """An independent sample from the same distribution becomes ≈ U(0, 1)."""

    rng    = np.random.default_rng(3)
    train  = rng.normal(size=10_000)
    test   = rng.normal(size=10_000)

    u = fit_uniformize(train)(test)

    stat, _ = kstest(u, "uniform")
    assert stat < 0.03


def test_uniformize_empty_train_raises():
    with pytest.raises(ValueError, match="empty"):
        fit_uniformize(np.array([]))


def test_uniformize_accepts_python_list():
    """Train and input may be plain Python sequences."""

    transform = fit_uniformize([1.0, 2.0, 3.0, 4.0, 5.0])
    u = transform([0.0, 3.0, 6.0])

    assert u.shape == (3,)
    assert (u > 0).all() and (u < 1).all()


# ---------------------------------------------------------------------------
# to_normal
# ---------------------------------------------------------------------------


def test_to_normal_zero_and_one_are_finite():
    """Boundary inputs must not produce ±inf — they should be clipped."""

    z = to_normal(np.array([0.0, 1.0]))

    assert np.isfinite(z).all()


def test_to_normal_half_maps_to_zero():
    assert to_normal(0.5) == pytest.approx(0.0)


def test_to_normal_is_antisymmetric_around_half():
    """probit(1 - u) == -probit(u)."""

    u = np.array([0.1, 0.2, 0.4, 0.45])
    np.testing.assert_allclose(to_normal(1 - u), -to_normal(u))


def test_to_normal_matches_scipy_norm_ppf_on_interior():
    u = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(to_normal(u), norm.ppf(u))


# ---------------------------------------------------------------------------
# composition — the headline behaviour
# ---------------------------------------------------------------------------


def test_pipeline_maps_arbitrary_marginal_to_standard_normal():
    """Uniformize + to_normal applied to an exponential sample is ≈ N(0, 1)."""

    rng    = np.random.default_rng(4)
    train  = rng.exponential(scale=2.0, size=20_000)
    test   = rng.exponential(scale=2.0, size=10_000)

    transform = fit_uniformize(train)
    z = to_normal(transform(test))

    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05

    stat, _ = kstest(z, "norm")
    assert stat < 0.02
