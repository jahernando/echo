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


# ---------------------------------------------------------------------------
# diagnose()
# ---------------------------------------------------------------------------


def test_diagnose_returns_expected_structure():
    train = make_sample(2_000, ["normal", "uniform", "normal"], seed=30)
    echo  = Echo()
    z, _  = echo.train(train)

    report = echo.diagnose(z)

    assert set(report) == {"marginals", "spearman"}
    assert list(report["marginals"].columns) == [
        "mean", "std", "skew", "excess_kurtosis", "ks_stat", "ks_pvalue",
    ]
    assert list(report["marginals"].index) == ["z0", "z1", "z2"]
    assert report["spearman"].shape == (3, 3)


def test_diagnose_marginals_on_h0_are_close_to_n01():
    train = make_sample(20_000, ["normal", "uniform", "normal"], correlation=0.5, seed=31)
    echo  = Echo()
    z, _  = echo.train(train)

    m = echo.diagnose(z)["marginals"]
    assert np.all(np.abs(m["mean"])              < 0.05)
    assert np.all(np.abs(m["std"] - 1.0)         < 0.05)
    assert np.all(np.abs(m["skew"])              < 0.10)
    assert np.all(np.abs(m["excess_kurtosis"])   < 0.20)
    assert np.all(m["ks_stat"]                   < 0.02)


def test_diagnose_spearman_on_h0_is_near_zero():
    """Even though Pearson is identity by construction, Spearman should also be ≈ 0 on H0."""

    train = make_sample(20_000, ["normal", "uniform", "normal"], correlation=0.5, seed=32)
    echo  = Echo()
    z, _  = echo.train(train)

    s = echo.diagnose(z)["spearman"].to_numpy()
    off_diag = s[~np.eye(3, dtype=bool)]
    assert np.all(np.abs(off_diag) < 0.03)


def test_diagnose_deep_returns_iterated_eigenvalues():
    train = make_sample(20_000, ["normal", "uniform", "normal"], seed=33)
    echo  = Echo()
    z, _  = echo.train(train)

    report = echo.diagnose(z, deep=True)

    assert "iterated_eigenvalues" in report
    assert report["iterated_eigenvalues"].shape == (3,)
    np.testing.assert_allclose(report["iterated_eigenvalues"], 1.0, atol=0.05)


def test_diagnose_deep_flags_non_gaussian_joint_structure():
    """If z has non-linear dependence, iterated eigenvalues spread away from 1."""

    rng = np.random.default_rng(34)
    x   = rng.standard_normal(20_000)
    y   = x ** 2 - 1                          # quadratic dependence — not captured by Pearson
    n   = rng.standard_normal(20_000)
    df  = pd.DataFrame({"x": x, "y": y, "z": n})

    echo = Echo()
    z, _ = echo.train(df)

    eigvals  = echo.diagnose(z, deep=True)["iterated_eigenvalues"]
    spread   = eigvals.max() - eigvals.min()
    assert spread > 0.1


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


def test_compare_returns_expected_structure():
    train = make_sample(2_000, ["normal", "uniform", "normal"], seed=40)
    test  = make_sample(1_000, ["normal", "uniform", "normal"], seed=41)
    echo  = Echo()
    echo.train(train)

    result = echo.compare(test)

    assert set(result) == {"z", "p", "marginals", "global"}
    assert result["z"].shape == (1_000, 3)
    assert result["p"].shape == (1_000,)
    assert list(result["marginals"].columns) == ["ks_stat", "ks_pvalue"]
    assert list(result["marginals"].index)   == ["z0", "z1", "z2"]
    assert set(result["global"]) == {
        "mean_p", "ks_stat_uniform", "ks_pvalue_uniform", "frac_below_alpha",
    }


def test_compare_on_h0_does_not_reject_uniformity():
    train = make_sample(20_000, ["normal", "uniform", "normal"], correlation=0.4, seed=42)
    test  = make_sample(10_000, ["normal", "uniform", "normal"], correlation=0.4, seed=43)

    echo = Echo()
    echo.train(train)
    result = echo.compare(test)

    assert result["global"]["ks_pvalue_uniform"]   > 0.05
    assert abs(result["global"]["mean_p"] - 0.5)   < 0.02

    # Per-component KS pvalues should be uniformly distributed; no extreme rejections.
    assert result["marginals"]["ks_pvalue"].min() > 0.01


def test_compare_detects_shift():
    train = make_sample(20_000, ["normal", "uniform", "normal"], correlation=0.4, seed=44)
    test  = make_sample(10_000, ["normal", "uniform", "normal"], correlation=0.4, seed=45)
    test["x"] = test["x"] + 0.5

    echo = Echo()
    echo.train(train)
    result = echo.compare(test)

    # Global uniformity is rejected and low-p tail shows excess.
    assert result["global"]["ks_pvalue_uniform"] < 1e-3
    assert result["global"]["frac_below_alpha"][0.05] > 0.06

    # Per-component KS rejects on at least one z_i.
    assert (result["marginals"]["ks_pvalue"] < 1e-3).any()


def test_compare_alphas_argument_is_respected():
    train = make_sample(2_000, ["normal", "uniform"], seed=46)
    test  = make_sample(1_000, ["normal", "uniform"], seed=47)
    echo  = Echo()
    echo.train(train)

    result = echo.compare(test, alphas=(0.001, 0.5))
    assert list(result["global"]["frac_below_alpha"].index) == [0.001, 0.5]


def test_compare_before_train_raises():
    test = make_sample(100, ["normal", "uniform"], seed=48)
    with pytest.raises(RuntimeError, match="not been trained"):
        Echo().compare(test)


def test_z_train_is_cached_after_train():
    train = make_sample(500, ["normal", "uniform"], seed=49)
    echo  = Echo()
    z, _  = echo.train(train)

    assert echo._z_train is not None
    pd.testing.assert_frame_equal(echo._z_train, z)


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
