"""The Echo two-sample comparison pipeline.

This module wires the per-variable transforms (``echo.transform``) into a
single stateful object that implements steps 1–8 of the algorithm:

    1. Uniformize each input variable using its ECDF on the train sample.
    2. Map each uniform variable to a standard normal via the probit.
    3. Find the symmetry axes (PCA) of the resulting cloud on the train.
    4. Rotate the variables onto those axes.
    5. Whiten by dividing each component by sqrt(eigenvalue).
    6. (Diagnostic) Check that the whitened variables are jointly N(0, I) —
       implemented as a separate method ``diagnose(z)``, not part of the
       train/test transform.
    7. For each event, compute chi2 = sum_i z_i^2.
    8. Build the ECDF of the train's chi2 and use ``p = 1 − ECDF(chi2)`` as
       the per-event p-value (physics convention: small p → anomalous, i.e.
       events far from the origin in whitened space). Under H0 the p-values
       are uniform on (0, 1).
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, kstest, kurtosis, skew

from echo.transform import fit_uniformize, to_normal


class Echo:
    """Two-sample comparison via per-variable normalization, whitening and chi2.

    Usage
    -----
    >>> echo = Echo()
    >>> z_train, p_train = echo.train(train_df)
    >>> z_test,  p_test  = echo.test(test_df)
    >>> report = echo.diagnose(z_train)
    >>> stats  = echo.compare(test_df)
    """

    def __init__(self):
        self._columns      = None    # list[str], set in train()
        self._uniformizers = None    # dict[str, callable]
        self._rotation     = None    # (d, d) ndarray
        self._eigenvalues  = None    # (d,)   ndarray
        self._scales       = None    # (d,)   ndarray, = sqrt(eigenvalues)
        self._chi2_pvalue  = None    # callable: chi2 -> p
        self._z_train      = None    # DataFrame (n_train, d), cached for compare()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def train(self, sample):
        """Fit the pipeline on a sample and return its transformed view.

        Parameters
        ----------
        sample : pandas.DataFrame or array_like, shape (n, d)
            The reference sample. Used to fit the per-column ECDFs, the
            rotation, the whitening scales and the chi2 ECDF.

        Returns
        -------
        z : pandas.DataFrame, shape (n, d)
            The whitened variables, columns ``z0 … z{d-1}``.
        p : pandas.Series, shape (n,)
            Per-row p-values from the train chi2 ECDF. Approximately uniform
            on (0, 1) by construction.
        """

        sample        = pd.DataFrame(sample).copy()
        self._columns = list(sample.columns)

        self._uniformizers = {c: fit_uniformize(sample[c]) for c in self._columns}

        z_pre = self._normalize_columns(sample)

        cov              = np.atleast_2d(np.cov(z_pre, rowvar=False))
        eigvals, eigvecs = np.linalg.eigh(cov)

        order             = np.argsort(eigvals)[::-1]
        self._eigenvalues = eigvals[order]
        self._rotation    = eigvecs[:, order]
        self._scales      = np.sqrt(self._eigenvalues)

        z, chi2           = self._rotate_whiten_chi2(z_pre)
        self._chi2_pvalue = fit_uniformize(chi2)

        z_df, p_series = self._wrap_outputs(z, chi2, sample.index)
        self._z_train  = z_df
        return z_df, p_series

    def compare(self, sample, alphas=(0.01, 0.05, 0.10)):
        """Apply the pipeline to ``sample`` and compare its distribution to the train.

        Parameters
        ----------
        sample : pandas.DataFrame or array_like
            Test sample to compare against the cached train.
        alphas : sequence of float, default (0.01, 0.05, 0.10)
            Significance thresholds for the low-p tail accounting.

        Returns
        -------
        dict
            ``"z"``, ``"p"`` : the transformed test and its p-values (same as
            ``test(sample)``).

            ``"marginals"`` : DataFrame indexed by ``z_i`` with the two-sample
            KS train-vs-test statistic and p-value per whitened component.
            Localizes the discrepancy in z-space.

            ``"global"`` : dict with ``mean_p``, ``ks_stat_uniform``,
            ``ks_pvalue_uniform`` (KS of the test p-values against U(0, 1))
            and ``frac_below_alpha`` (Series indexed by ``alphas``).
        """

        self._require_trained()
        z, p = self.test(sample)

        rows = []
        for col in self._z_train.columns:
            ks = ks_2samp(self._z_train[col].to_numpy(), z[col].to_numpy())
            rows.append({"ks_stat": ks.statistic, "ks_pvalue": ks.pvalue})
        marginals = pd.DataFrame(rows, index=list(self._z_train.columns))

        ks_uniform   = kstest(p.to_numpy(), "uniform")
        frac_low_p   = pd.Series(
            {float(a): float((p < a).mean()) for a in alphas},
            name="frac_below_alpha",
        )

        global_stats = {
            "mean_p":            float(p.mean()),
            "ks_stat_uniform":   float(ks_uniform.statistic),
            "ks_pvalue_uniform": float(ks_uniform.pvalue),
            "frac_below_alpha":  frac_low_p,
        }

        return {
            "z":         z,
            "p":         p,
            "marginals": marginals,
            "global":    global_stats,
        }

    def diagnose(self, z, deep=False):
        """Diagnose Gaussianity / decorrelation quality of a transformed sample.

        Parameters
        ----------
        z : pandas.DataFrame, shape (n, d)
            The output of ``train(...)`` or ``test(...)`` — variables already
            in the whitened ``z``-space.
        deep : bool, default False
            If True, also run an *iterated* check: re-apply per-column
            uniformize+probit to ``z`` and rerun PCA. The resulting eigenvalue
            spectrum should be all ≈ 1 if ``z`` is jointly Gaussian. Spread
            away from 1 indicates non-linear residual dependence that PCA
            alone cannot remove.

        Returns
        -------
        dict
            ``"marginals"`` : DataFrame indexed by column of ``z`` with
            ``mean``, ``std``, ``skew``, ``excess_kurtosis``, ``ks_stat``,
            ``ks_pvalue`` (KS against N(0, 1)).

            ``"spearman"`` : DataFrame ``(d × d)`` of Spearman (rank)
            correlations. Pearson on the train is identity by construction
            so we report Spearman, which captures monotone non-linear
            dependence PCA does not remove.

            ``"iterated_eigenvalues"`` : ndarray ``(d,)``, only present if
            ``deep=True``.
        """

        rows = []
        for col in z.columns:
            values   = z[col].to_numpy()
            ks       = kstest(values, "norm")
            rows.append({
                "mean":            values.mean(),
                "std":             values.std(ddof=1),
                "skew":            skew(values),
                "excess_kurtosis": kurtosis(values, fisher=True),
                "ks_stat":         ks.statistic,
                "ks_pvalue":       ks.pvalue,
            })
        marginals = pd.DataFrame(rows, index=z.columns)

        result = {
            "marginals": marginals,
            "spearman":  z.corr(method="spearman"),
        }
        if deep:
            result["iterated_eigenvalues"] = self._iterated_eigenvalues(z)
        return result

    def test(self, sample):
        """Apply the fitted pipeline to a new sample.

        Parameters
        ----------
        sample : pandas.DataFrame or array_like, shape (n, d)
            Must have the same columns (in the same order) as the train sample.

        Returns
        -------
        z : pandas.DataFrame, shape (n, d)
        p : pandas.Series, shape (n,)
        """

        self._require_trained()
        sample = pd.DataFrame(sample).copy()

        missing = [c for c in self._columns if c not in sample.columns]
        if missing:
            raise ValueError(f"test sample is missing columns: {missing}")
        sample = sample[self._columns]

        z_pre   = self._normalize_columns(sample)
        z, chi2 = self._rotate_whiten_chi2(z_pre)
        return self._wrap_outputs(z, chi2, sample.index)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _normalize_columns(self, sample):
        """Apply steps 1–2 (ECDF + probit) per column. Returns (n, d) ndarray."""

        out = np.empty((len(sample), len(self._columns)), dtype=float)
        for i, col in enumerate(self._columns):
            u         = self._uniformizers[col](sample[col].to_numpy())
            out[:, i] = to_normal(u)
        return out

    def _rotate_whiten_chi2(self, z_pre):
        """Apply steps 4, 5 and 7 with the fitted rotation and scales."""

        z    = (z_pre @ self._rotation) / self._scales
        chi2 = (z ** 2).sum(axis=1)
        return z, chi2

    def _wrap_outputs(self, z, chi2, index):
        """Build the user-facing (DataFrame, Series) return values."""

        z_df = pd.DataFrame(
            z,
            index   = index,
            columns = [f"z{i}" for i in range(z.shape[1])],
        )
        # Physics convention: small p → far-from-origin (anomalous) event.
        p = 1.0 - self._chi2_pvalue(chi2)
        return z_df, pd.Series(p, index=index, name="p_value")

    def _iterated_eigenvalues(self, z):
        """Re-Gaussianize per column, then PCA: eigenvalues ≈ 1 ⇒ joint N(0, I)."""

        z_renorm = np.column_stack([to_normal(fit_uniformize(z[c])(z[c])) for c in z.columns])
        cov      = np.atleast_2d(np.cov(z_renorm, rowvar=False))
        eigvals  = np.linalg.eigvalsh(cov)
        return np.sort(eigvals)[::-1]

    def _require_trained(self):
        if self._rotation is None:
            raise RuntimeError("Echo has not been trained yet; call .train(...) first")
