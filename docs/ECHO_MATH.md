# The Echo algorithm — mathematical description

`echo` compares two samples by **(a)** transforming the variables of a
reference (train) sample to an approximately standard normal, uncorrelated
distribution, and **(b)** scoring any other sample in that transformed space.
The headline outputs are a per-event $\chi^2$ and a calibrated p-value with
respect to the train.

This document is the formal companion to the implementation. See
`BIBLIS.md` §10 for the literature context (Iterative Gaussianization, PIT,
Mahalanobis, Nonparanormal).

---

## 1. Setup and notation

Let $X^{(\mathrm{train})} \in \mathbb{R}^{n_{\mathrm{tr}} \times d}$ and
$X^{(\mathrm{test})} \in \mathbb{R}^{n_{\mathrm{te}} \times d}$ be two samples,
each a set of i.i.d. draws of a $d$-dimensional continuous random vector. We
denote the $k$-th event of a sample by a row vector $x_k \in \mathbb{R}^d$ and
its $j$-th component by $x_{k,j}$.

Two questions:

1. **Same distribution?** Are train and test drawn from the same law?
2. **Where do they differ?** If not, which regions of $\mathbb{R}^d$ are
   responsible for the discrepancy?

`echo` answers both with a single pipeline.

---

## 2. The transformation $\varphi$ — steps 1–5

Steps 1–5 build a map $\varphi : \mathbb{R}^d \to \mathbb{R}^d$, fitted on the
train, such that $\varphi(X^{(\mathrm{train})}) \approx \mathcal{N}(0, I_d)$.
The same $\varphi$ is then applied to any other sample.

### 2.1 Step 1 — marginal uniformization

For each component $j \in \{1, \dots, d\}$, let $\hat F_j$ be the empirical
CDF of column $j$ on the train, evaluated with the mid-rank convention:

$$
\hat F_j(x) \;=\; \frac{ \#\{k : X^{(\mathrm{train})}_{k,j} \le x\} \;+\; \tfrac{1}{2} }{ n_{\mathrm{tr}} + 1 }.
$$

Define $u_j = \hat F_j(x_j)$. By construction $u_j \in
\bigl(\tfrac{1/2}{n_{\mathrm{tr}}+1},\, \tfrac{n_{\mathrm{tr}}+1/2}{n_{\mathrm{tr}}+1}\bigr) \subset (0, 1)$
for **any** input, including values outside the train range.

By the **Probability Integral Transform** (Rosenblatt 1952), if $X_j$ has
continuous CDF $F_j$, then $F_j(X_j) \sim \mathcal{U}(0, 1)$. The ECDF $\hat F_j$
approximates $F_j$, so on a fresh sample drawn from the same law,
$\hat F_j(X_j)$ is approximately uniform.

### 2.2 Step 2 — probit

Apply the inverse standard-normal CDF $\Phi^{-1}$ component-wise:

$$
\tilde z_j \;=\; \Phi^{-1}(u_j).
$$

Composed with step 1: $\tilde z_j$ is approximately $\mathcal{N}(0, 1)$
marginally. In implementation, $u_j$ is clipped to
$(\varepsilon, 1 - \varepsilon)$ with $\varepsilon = 10^{-12}$ to avoid
$\pm\infty$ at the boundaries.

At this point each marginal is approximately standard normal, but the
**joint** distribution is generally **not** $\mathcal{N}(0, I_d)$: the
correlation structure of the original variables survives the marginal
mapping (this is exactly the *Gaussian copula* construction).

### 2.3 Step 3 — symmetry axes (PCA)

Compute the sample covariance of $\tilde z$ on the train:

$$
\hat\Sigma \;=\; \frac{1}{n_{\mathrm{tr}} - 1} \sum_{k=1}^{n_{\mathrm{tr}}} \bigl(\tilde z_k - \bar{\tilde z}\bigr) \bigl(\tilde z_k - \bar{\tilde z}\bigr)^{\!\top}.
$$

Eigendecompose:

$$
\hat\Sigma \;=\; V\, \Lambda\, V^{\!\top}, \qquad \Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_d),\quad \lambda_1 \ge \dots \ge \lambda_d > 0,
$$

with $V \in O(d)$ orthogonal. The columns of $V$ are the *symmetry axes* —
the principal directions of variation in the marginally-Gaussianized space.

### 2.4 Step 4 — rotation

$$
Z \;=\; \tilde z\, V.
$$

The rotated coordinates have diagonal covariance $\Lambda$. They are
**linearly** decorrelated; non-linear dependence is untouched (PCA cannot
remove it).

### 2.5 Step 5 — whitening

$$
z \;=\; Z\, \Lambda^{-1/2}, \quad \text{i.e.,} \quad z_j = \frac{Z_j}{\sqrt{\lambda_j}}.
$$

By construction $\mathrm{Cov}(z^{(\mathrm{train})}) = I_d$ **exactly**.

The full forward map is

$$
\varphi(x) \;=\; \Lambda^{-1/2}\, V^{\!\top}\, \bigl(\Phi^{-1} \circ \hat F\bigr)(x),
$$

where $\hat F = (\hat F_1, \dots, \hat F_d)$ acts componentwise.

#### Jacobian

For density-based extensions (proper likelihood ratio, see §6), the Jacobian
factorizes:

$$
\bigl|\det \nabla \varphi(x)\bigr| \;=\; \frac{1}{\sqrt{\det \hat\Sigma}} \prod_{j=1}^{d} \frac{ \hat f_j(x_j) }{ \varphi\!\bigl(\tilde z_j\bigr) },
$$

with $\hat f_j$ the (numerical or analytic) derivative of $\hat F_j$ and
$\varphi(\cdot)$ the standard-normal pdf. This is not used by the basic
pipeline; reported here for completeness.

---

## 3. Step 6 — diagnostics

After step 5 the train $z$ has covariance exactly $I_d$. The question is
whether it is also **jointly Gaussian**. Three checks:

1. **Marginals.** For each $z_j$ compute mean, std, skewness, excess
   kurtosis, and a one-sample Kolmogorov–Smirnov statistic against
   $\mathcal{N}(0, 1)$.
2. **Spearman correlation matrix.** Captures monotone non-linear dependence
   that PCA cannot remove. (Pearson on the train is trivially $I_d$ by step 5,
   hence uninformative.)
3. **Iterated eigenvalues (deep).** Re-apply marginal uniformize + probit +
   PCA to the already-whitened $z$; the resulting eigenvalues should be all
   $\approx 1$ if and only if $z$ is jointly Gaussian. Spread away from 1
   quantifies residual non-Gaussianity. This is equivalent to checking that
   one iteration of the algorithm was a near-fixed-point of the RBIG
   iteration (Laparra–Camps-Valls–Malo 2011).

---

## 4. Steps 7–8 — chi-squared and p-value

### 4.1 Step 7 — chi-squared per event

For an event $x$ with whitened image $z = \varphi(x) \in \mathbb{R}^d$:

$$
\chi^2(x) \;=\; \|z\|^2 \;=\; \sum_{j=1}^{d} z_j^2.
$$

This is the **Mahalanobis distance squared** of $x$ from the center of the
train cloud, measured in the metric induced by $\hat\Sigma$ on the
Gaussianized variables. Under perfect joint Gaussianity, $\chi^2 \sim
\chi^2_d$.

### 4.2 Step 8 — calibrated p-value

Rather than relying on the theoretical $\chi^2_d$ distribution (which would
require exact joint Gaussianity), `echo` calibrates against the train itself.
Let $\{\chi^2_k\}_{k=1}^{n_{\mathrm{tr}}}$ be the chi-squared values of the
training events. Define their ECDF (mid-rank convention):

$$
\hat F_{\chi^2}(t) \;=\; \frac{ \#\{k : \chi^2_k \le t\} + \tfrac{1}{2} }{ n_{\mathrm{tr}} + 1 }.
$$

The per-event p-value is (physics convention)

$$
p(x) \;=\; 1 \;-\; \hat F_{\chi^2}\!\bigl(\chi^2(x)\bigr).
$$

Sign convention: $p$ small $\Leftrightarrow$ $\chi^2$ large $\Leftrightarrow$
event far from the origin in whitened space $\Leftrightarrow$ anomalous.

#### Key properties

- **Self-uniformity.** When applied to the train itself, $p_{\mathrm{train}}$
  is exactly uniform on a discrete grid of mid-ranks — it is the textbook
  identity $\hat F(X) \sim \mathcal{U}$ for a sample under its own ECDF, valid
  for *any* underlying distribution (no Gaussianity required).
- **Validity under H₀.** If $X^{(\mathrm{test})}$ comes from the same
  distribution as the train, then $\chi^2(X^{(\mathrm{test})})$ has the same
  distribution as $\chi^2(X^{(\mathrm{train})})$, hence
  $p(X^{(\mathrm{test})}) \sim \mathcal{U}(0,1)$ approximately.
- **Power under H₁.** Departures from H₀ shift the $\chi^2$ distribution and
  thus push $p$ away from uniformity. The departure can be tested by, e.g.,
  a KS test of $p^{(\mathrm{test})}$ against $\mathcal{U}(0, 1)$.

---

## 5. Two-sample comparison: `compare`

Given a test sample, compute:

1. The transformation: $z^{(\mathrm{test})} = \varphi(X^{(\mathrm{test})})$,
   $p^{(\mathrm{test})}$.
2. **Per-component two-sample KS** between $z_j^{(\mathrm{train})}$ and
   $z_j^{(\mathrm{test})}$ for $j = 1, \dots, d$. Localizes the discrepancy in
   the whitened space.
3. **Global uniformity of $p^{(\mathrm{test})}$**: one-sample KS against
   $\mathcal{U}(0, 1)$, plus $\frac{1}{n_{\mathrm{te}}}\#\{k : p_k < \alpha\}$
   for several thresholds $\alpha$.

Under H₀ all of these are statistically null; under H₁ at least the global
KS rejects, and the per-component KS picks up which whitened directions
carry the discrepancy.

---

## 6. Two `Echo` instances: H₁ vs H₀ discrimination

Train one `Echo` on samples from H₀ and another on samples from H₁. For any
event $x$ define

$$
\Delta\chi^2(x) \;=\; \chi^2_{H_0}(x) \;-\; \chi^2_{H_1}(x).
$$

#### Interpretation

Under joint Gaussianity in each whitened space, the $H$-density at $x$
factorizes as

$$
p_H(x) \;=\; (2\pi)^{-d/2}\, \exp\!\bigl(-\tfrac{1}{2}\chi^2_H(x)\bigr) \;\cdot\; \bigl|\det \nabla \varphi_H(x)\bigr|,
$$

where $\varphi_H$ is the $H$-fitted transform of §2.5. Taking logs,

$$
-2 \log p_H(x) \;=\; \chi^2_H(x) \;+\; d\log(2\pi) \;-\; 2 \log\bigl|\det \nabla \varphi_H(x)\bigr|.
$$

Subtracting between $H_1$ and $H_0$,

$$
\Delta\chi^2(x) \;=\; 2 \log \frac{ p_{H_1}(x) }{ p_{H_0}(x) } \;+\; 2 \log \frac{ |\det \nabla \varphi_{H_1}(x)| }{ |\det \nabla \varphi_{H_0}(x)| }.
$$

The first term on the right is the (signed) log-likelihood ratio one wants;
the second is a **residual Jacobian correction** that is in general
$x$-dependent — it cancels exactly only when $H_0$ and $H_1$ share identical
marginal CDFs and identical principal-axes geometry. Sign convention:

| sign | meaning |
|---|---|
| $\Delta\chi^2 > 0$ | event more H₁-like |
| $\Delta\chi^2 < 0$ | event more H₀-like |

#### Limitations and faithfulness

When $H_0$ and $H_1$ are close (e.g. same marginal shapes, similar
correlation structure), the Jacobian correction is small and $\Delta\chi^2$
is a faithful proxy of the LR. When marginal forms differ strongly, the
residual is non-negligible and $\Delta\chi^2$ is a *biased* (though usually
still monotonic) summary. The fully calibrated density-based LR — adding
the explicit PIT derivatives and the $|\hat\Sigma|^{-1/2}$ factor of §2.5 —
is left as a future extension.

---

## 7. Computational cost

| step | cost (fit) | cost (apply, size $n$) |
|---|---|---|
| ECDF per column | $O(n_{\mathrm{tr}} \log n_{\mathrm{tr}})$ | $O(n \log n_{\mathrm{tr}})$ via binary search |
| probit | $O(n_{\mathrm{tr}})$ | $O(n)$ |
| covariance + eigh | $O(n_{\mathrm{tr}} d^2 + d^3)$ | — |
| rotation + whitening | — | $O(n d^2)$ |
| $\chi^2$ per event | $O(n_{\mathrm{tr}} d)$ | $O(n d)$ |
| $\chi^2$ ECDF | $O(n_{\mathrm{tr}} \log n_{\mathrm{tr}})$ | $O(n \log n_{\mathrm{tr}})$ |

The pipeline is linear in $n$ and at worst $O(d^3)$ in dimension (the eigh
on $\hat\Sigma$). For $d$ in the tens this is essentially free; for $d$ in
the hundreds, switch to a truncated SVD if needed.

---

## 8. Relation to existing methods

Short cross-reference with the closest matches in the literature. Full
bibliography (with all sections) in `BIBLIS.md`.

### 8.1 Probability Integral Transform (Rosenblatt)

Rosenblatt, M. *Remarks on a multivariate transformation.* Ann. Math. Stat.
23(3):470–472 (1952).

**Relation.** Foundation of step 1: if $X_j$ has continuous CDF $F_j$, then
$F_j(X_j) \sim \mathcal{U}(0, 1)$. In `echo` the population CDF is replaced
by its empirical version $\hat F_j$.

**Implementation building blocks**:
[`scipy.stats.rankdata`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html),
[`scipy.stats.norm.ppf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
(probit),
[`numpy.searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html).

### 8.2 Iterative Gaussianization / RBIG

Chen, S. S., Gopinath, R. A. *Gaussianization.* NeurIPS 2000. — Laparra, V.,
Camps-Valls, G., Malo, J. *Iterative Gaussianization: From ICA to Random
Rotations.* IEEE Trans. Neural Netw. 22(4):537–549 (2011).
[arXiv:1004.0925](https://arxiv.org/abs/1004.0925).

**Relation.** RBIG iterates two steps until convergence: (a) marginal
Gaussianization via PIT + probit; (b) orthogonal rotation (PCA, ICA, or
random). One iteration of RBIG with PCA is **exactly steps 1–4** of `echo`.
`echo` then adds explicit whitening (step 5) and, instead of iterating to
exact joint Gaussianity, absorbs the residual non-Gaussianity into an
empirical $\chi^2$ ECDF (step 8). `Echo.diagnose(z, deep=True)` checks how
close `echo` is to an RBIG fixed point in one shot.

### 8.3 Nonparanormal (NPN) / Gaussian copula

Liu, H., Lafferty, J., Wasserman, L. *The Nonparanormal: Semiparametric
Estimation of High Dimensional Undirected Graphs.* JMLR 10:2295–2328 (2009).
[arXiv:0903.0649](https://arxiv.org/abs/0903.0649). Classical foundation:
Sklar, A. *Fonctions de répartition à n dimensions et leurs marges.* Publ.
Inst. Statist. Univ. Paris 8:229–231 (1959).

**Relation.** Models $X = f(Y)$ with $Y \sim \mathcal{N}(\mu, \Sigma)$ and
$f$ a vector of monotone marginal functions. The marginal map is identical
to steps 1–2 of `echo`. NPN's downstream goal is to recover the latent
covariance / graphical model in $Y$-space; `echo` whitens that latent and
turns the per-event squared norm into a calibrated p-value.

### 8.4 Hotelling $T^2$ / Mahalanobis distance

Hotelling, H. *The generalization of Student's ratio.* Ann. Math. Stat.
2(3):360–378 (1931). — Mahalanobis, P. C. *On the generalised distance in
statistics.* Proc. Natl. Inst. Sci. India 2(1):49–55 (1936).

**Relation.** For $X \sim \mathcal{N}(\mu, \Sigma)$, the statistic $T^2 =
(X-\mu)^\top \Sigma^{-1}(X-\mu)$ is distributed $\chi^2_d$ (or $F$ for
estimated $\Sigma$). After `echo`'s steps 1–5, $\chi^2(x) = \|z\|^2$ is
**exactly** the squared Mahalanobis distance in the whitened space. The
distinctive feature of `echo` is the *empirical* calibration via
$\hat F_{\chi^2}$ on the train, valid without assuming the original
variables are Gaussian.

**Implementation building block**:
[`scipy.spatial.distance.mahalanobis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html).

### 8.5 Other useful comparators

For two-sample testing in general (not necessarily Gaussianization-based),
see `BIBLIS.md` sections 1–9. Particularly relevant comparators when
benchmarking `echo`:

- **Two-sample Kolmogorov–Smirnov** (per dimension or on the p-value
  distribution): [`scipy.stats.ks_2samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).
- **Energy distance / Aslan–Zech** (distribution-free, multivariate,
  $O(n^2)$): Székely & Rizzo (2013); Aslan & Zech (2005). No arXiv, see
  `BIBLIS.md` §2.2 and §7.1.
- **Maximum Mean Discrepancy (MMD)**: Gretton et al. JMLR 13:723–773 (2012),
  [arXiv:0805.2368](https://arxiv.org/abs/0805.2368). The kernel two-sample
  test. Witness function gives localization.
- **Classifier Two-Sample Test (C2ST)**: Lopez-Paz & Oquab. ICLR 2017,
  [arXiv:1610.06545](https://arxiv.org/abs/1610.06545). Trains a classifier
  on the joined sample; reads off accuracy as the statistic.
- **NPLM (HEP goodness-of-fit by Neyman–Pearson)**: Grosso, Letizia, Pierini,
  Wulzer. SciPost Phys. 16:123 (2024). Related in spirit to `score_lr` but
  built around a regularized classifier rather than two Gaussianizers.
