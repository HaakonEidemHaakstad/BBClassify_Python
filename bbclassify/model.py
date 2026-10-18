"""The beta-binomial model and its extensions.

Notation follows the BB-CLASS manual (Brennan, 2004):

* ``n``   number of (effective) dichotomous items, so observed scores are 0..n
* ``tau`` proportion-correct true score, distributed as a two- or
  four-parameter beta on [l, u]
* ``k``   Lord's k; k = 0 gives the binomial error model, k > 0 Lord's (1965)
  two-term approximation to the compound binomial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

from .feedback import BBClassifyError, Feedback

# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Moments:
    """Mean, standard deviation, skewness and kurtosis (not excess kurtosis).

    Variances use the number of examinees as divisor, as in BB-CLASS.
    """

    mean: float
    sd: float
    skew: float
    kurt: float

    @property
    def var(self) -> float:
        return self.sd ** 2

    def power_moments(self) -> np.ndarray:
        """E[X], E[X^2], E[X^3], E[X^4]."""
        m, v = self.mean, self.var
        c3 = self.skew * v ** 1.5
        c4 = self.kurt * v ** 2
        return np.array([m,
                         v + m ** 2,
                         c3 + 3 * m * v + m ** 3,
                         c4 + 4 * m * c3 + 6 * m ** 2 * v + m ** 4])

    def rescaled(self, scale: float, shift: float = 0.0) -> "Moments":
        """Moments of ``scale * (X - shift)``; skewness and kurtosis are unchanged."""
        return Moments((self.mean - shift) * scale, self.sd * abs(scale), self.skew, self.kurt)


def distribution_moments(scores: np.ndarray, weights: np.ndarray) -> Moments:
    """Moments of a (frequency-weighted) score distribution."""
    w = weights / weights.sum()
    mean = float(np.sum(w * scores))
    dev = scores - mean
    var = float(np.sum(w * dev ** 2))
    if var <= 0:
        raise BBClassifyError(
            "All examinees have the same score, so the score variance is zero. "
            "Classification consistency and accuracy cannot be estimated from such data.")
    return Moments(mean, math.sqrt(var),
                   float(np.sum(w * dev ** 3)) / var ** 1.5,
                   float(np.sum(w * dev ** 4)) / var ** 2)


def central_from_raw(raw: np.ndarray) -> Moments:
    """Mean, SD, skewness and kurtosis from the first four raw (power) moments."""
    m1, m2, m3, m4 = raw
    var = m2 - m1 ** 2
    if not var > 0:
        return Moments(m1, float("nan"), float("nan"), float("nan"))
    c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return Moments(m1, math.sqrt(var), c3 / var ** 1.5, c4 / var ** 2)


def lords_k(mean: float, var: float, reliability: float, n: int) -> float:
    """Lord's k from the observed mean and variance and a reliability coefficient."""
    true_var = var * reliability
    numerator = n * ((n - 1) * true_var - n * var + mean * (n - mean))
    denominator = 2 * (mean * (n - mean) - true_var)
    return numerator / denominator


def effective_test_length(mean: float, var: float, reliability: float,
                          xmin: float, xmax: float) -> float:
    """Livingston and Lewis' effective test length.

    ``xmin`` and ``xmax`` are the lowest and highest *possible* scores on the
    test. BB-CLASS used the lowest and highest *observed* scores, which is
    the error this program corrects.
    """
    return ((mean - xmin) * (xmax - mean) - reliability * var) / (var * (1 - reliability))


def true_score_moments(observed: Moments, n: int, k: float) -> np.ndarray:
    """First four raw moments of the proportion-correct true score.

    Lord (1965, eq. 37), via the observed factorial moments. ``observed`` are
    the moments of scores on the 0..n scale.
    """
    m1, m2, m3, m4 = observed.power_moments()
    factorial = [m1, m2 - m1, m3 - 3 * m2 + 2 * m1, m4 - 6 * m3 + 11 * m2 - 6 * m1]
    mu = np.zeros(4)
    mu[0] = m1 / n
    falling_n_minus_2 = 1.0  # (n - 2)^(r - 2), the falling factorial
    for r in range(2, 5):
        if r > 2:
            falling_n_minus_2 *= (n - r + 1)
        mu[r - 1] = (factorial[r - 1] / falling_n_minus_2 + k * r * (r - 1) * mu[r - 2]) / (n * (n - 1) + k * r * (r - 1))
    return mu


# ---------------------------------------------------------------------------
# Beta true-score distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaFit:
    alpha: float
    beta: float
    lower: float
    upper: float
    moments_fit: int

    def density(self, tau: np.ndarray) -> np.ndarray:
        width = self.upper - self.lower
        return stats.beta.pdf((tau - self.lower) / width, self.alpha, self.beta) / width

    def cdf(self, tau: np.ndarray) -> np.ndarray:
        return stats.beta.cdf((tau - self.lower) / (self.upper - self.lower), self.alpha, self.beta)

    def moments(self) -> Moments:
        """Moments of the true score on the proportion-correct scale."""
        a, b, w = self.alpha, self.beta, self.upper - self.lower
        s = a + b
        mean = self.lower + w * a / s
        var = w ** 2 * a * b / (s ** 2 * (s + 1))
        skew = 2 * (b - a) * math.sqrt(s + 1) / ((s + 2) * math.sqrt(a * b))
        excess = 6 * ((a - b) ** 2 * (s + 1) - a * b * (s + 2)) / (a * b * (s + 2) * (s + 3))
        return Moments(mean, math.sqrt(var), skew, excess + 3)


def _admissible(a: float, b: float, l: float, u: float) -> bool:
    values = (a, b, l, u)
    return (all(math.isfinite(v) for v in values) and a > 0 and b > 0
            and -1e-12 <= l < u <= 1 + 1e-12)


def _fit_four(t: Moments) -> tuple[float, float, float, float]:
    """Four-parameter method of moments (Hanson, 1991)."""
    g3, g4 = t.skew, t.kurt
    with np.errstate(all="ignore"):
        r = 6 * (g4 - g3 ** 2 - 1) / (6 + 3 * g3 ** 2 - 2 * g4)
        disc = 1 - 24 * (r + 1) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))
        if not (math.isfinite(r) and math.isfinite(disc) and disc >= 0):
            return (float("nan"),) * 4
        root = math.sqrt(disc)
        larger, smaller = r / 2 * (1 + root), r / 2 * (1 - root)
        a, b = (larger, smaller) if g3 < 0 else (smaller, larger)
        if not (a > 0 and b > 0):
            return (float("nan"),) * 4
        spread = math.sqrt(t.var * (a + b + 1)) / math.sqrt(a * b)
        return a, b, t.mean - a * spread, t.mean + b * spread


def _fit_three(t: Moments, fixed: str) -> tuple[float, float, float, float] | None:
    """Three-moment fit with one limit fixed (``"upper"`` at 1 or ``"lower"`` at 0).

    With p = alpha / (alpha + beta), the fixed limit and the mean and variance
    determine alpha + beta as a function of p, and the skewness equation is
    then solved for p.
    """
    if fixed == "upper":
        distance = 1.0 - t.mean
    else:
        distance = t.mean
    if not distance > 0:
        return None
    ratio = distance ** 2 / t.var

    def total(p: float) -> float:  # alpha + beta
        return (ratio * p / (1 - p) if fixed == "upper" else ratio * (1 - p) / p) - 1

    def excess_skew(p: float) -> float:
        s = total(p)
        if s <= 0:
            return float("nan")
        return 2 * (1 - 2 * p) * math.sqrt(s + 1) / ((s + 2) * math.sqrt(p * (1 - p))) - t.skew

    grid = np.linspace(1e-6, 1 - 1e-6, 4001)
    values = np.array([excess_skew(p) for p in grid])
    for i in range(len(grid) - 1):
        f0, f1 = values[i], values[i + 1]
        if not (math.isfinite(f0) and math.isfinite(f1)) or f0 * f1 > 0:
            continue
        p = optimize.brentq(excess_skew, grid[i], grid[i + 1], xtol=1e-15) if f0 * f1 < 0 else grid[i]
        s = total(p)
        a, b = p * s, (1 - p) * s
        if fixed == "upper":
            width = distance / (1 - p)
            candidate = (a, b, 1.0 - width, 1.0)
        else:
            width = distance / p
            candidate = (a, b, 0.0, width)
        if _admissible(*candidate):
            return candidate
    return None


def _fit_two(t: Moments) -> tuple[float, float, float, float] | None:
    m, v = t.mean, t.var
    common = m * (1 - m) / v - 1
    a, b = m * common, (1 - m) * common
    return (a, b, 0.0, 1.0) if _admissible(a, b, 0.0, 1.0) else None


def fit_beta(true_moments: Moments, nparm: int, feedback: Feedback) -> BetaFit:
    """Method-of-moments fit of the beta true-score distribution.

    With ``nparm = 4``, the four-parameter solution is used when admissible
    (0 <= l < u <= 1). Otherwise, as in BB-CLASS, three moments are fit with
    the violated limit fixed (u = 1 or l = 0), and if that also fails, the
    two-parameter beta (l = 0, u = 1) is fit to two moments.
    """
    t = true_moments
    if not (t.var > 0 and math.isfinite(t.var)):
        raise BBClassifyError(
            "The estimated true-score variance is not positive, so no beta true-score distribution can be fit. "
            "This usually means the reliability is too low for the observed score variance, "
            "or (for LL) that the effective test length is too short. Check the reliability on control card 1"
            " and, for LL, the possible minimum and maximum scores on control card 2.")
    if nparm == 4:
        four = _fit_four(t)
        if _admissible(*four):
            return BetaFit(*four, moments_fit=4)
        a, b, l, u = four
        if all(math.isfinite(v) for v in four):
            problem = f"(l = {l:.5f}, u = {u:.5f})"
        else:
            problem = "(no real-valued solution)"
        order = ["upper", "lower"] if not (math.isfinite(l) and l < 0 and u <= 1) else ["lower", "upper"]
        for fixed in order:
            three = _fit_three(t, fixed)
            if three is not None:
                limit = "upper limit u = 1" if fixed == "upper" else "lower limit l = 0"
                feedback.note(
                    f"The four-parameter beta solution was not admissible {problem}, so three moments were fit "
                    f"with the {limit} (as in BB-CLASS).")
                return BetaFit(*three, moments_fit=3)
        feedback.warn(
            f"Neither the four-parameter beta solution {problem} nor a three-moment solution was admissible, "
            "so the two-parameter beta (l = 0, u = 1) was fit to the mean and variance. "
            "Model fit may be poor; inspect the chi-square statistics.")
    two = _fit_two(t)
    if two is None:
        raise BBClassifyError(
            "No two-parameter beta distribution has the estimated true-score mean and variance "
            f"(mean = {t.mean:.5f}, variance = {t.var:.5f} on the proportion-correct scale). "
            "The true-score variance is too large relative to the mean; check the reliability and the input data.")
    return BetaFit(*two, moments_fit=2)


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------


def error_pmf(n: int, k: float, tau: np.ndarray) -> np.ndarray:
    """P(X = i | tau) for i = 0..n, as a (len(tau), n + 1) matrix.

    Binomial when k = 0; otherwise Lord's two-term approximation to the
    compound binomial.
    """
    tau = np.asarray(tau, dtype=float)[:, None]
    i = np.arange(n + 1)[None, :]
    pmf = stats.binom.pmf(i, n, tau)
    if k != 0:
        b = lambda j: stats.binom.pmf(j, n - 2, tau)  # noqa: E731
        pmf = pmf - k * tau * (1 - tau) * (b(i) - 2 * b(i - 1) + b(i - 2))
    return pmf


def marginal_pmf(fit: BetaFit, n: int, k: float) -> np.ndarray:
    """Exact model probabilities P(X = i), i = 0..n.

    The conditional error probabilities are polynomials of degree n in the
    true score, so Gauss-Jacobi quadrature with the beta density as weight
    function integrates them exactly.
    """
    x, w = special.roots_jacobi(n + 3, fit.beta - 1, fit.alpha - 1)
    tau = fit.lower + (fit.upper - fit.lower) * (x + 1) / 2
    return (w / w.sum()) @ error_pmf(n, k, tau)


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

QUADRATURE_METHODS = ("EDquad", "EHquad", "Gquad")
GAUSS_POINTS = 64


@dataclass(frozen=True)
class Quadrature:
    nodes: np.ndarray    # true-score values (proportion-correct)
    weights: np.ndarray  # probability mass attached to each node
    method: str
    npts: int

    def description(self) -> str:
        if self.method == "EDquad":
            return (f"Numerical integration performed using {self.npts} equally-spaced quadrature points\n"
                    "   and the true-score density for each interval")
        if self.method == "EHquad":
            return (f"Numerical integration performed using {self.npts} equally-spaced quadrature points\n"
                    "   and the proportional height of the true-score density at each point")
        return f"Numerical integration performed using {GAUSS_POINTS}-point Gauss quadrature"


def quadrature(fit: BetaFit, method: str, npts: int, true_cuts: np.ndarray) -> Quadrature:
    """Quadrature nodes and weights over the true-score range [l, u].

    Intervals that contain a true cut score are split at the cut so that the
    accuracy integrals are not distorted by intervals straddling a cut.
    """
    l, u = fit.lower, fit.upper
    splits = [c for c in true_cuts if l < c < u]
    if method == "Gquad":
        x, w = np.polynomial.legendre.leggauss(GAUSS_POINTS)
        bounds = np.concatenate([[l], splits, [u]])
        nodes, weights = [], []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            half = (hi - lo) / 2
            tau = lo + half * (x + 1)
            nodes.append(tau)
            weights.append(w * half * fit.density(tau))
        return Quadrature(np.concatenate(nodes), np.concatenate(weights), method, GAUSS_POINTS)
    edges = np.unique(np.concatenate([np.linspace(l, u, npts + 1), splits]))
    mids = (edges[:-1] + edges[1:]) / 2
    if method == "EDquad":
        weights = np.diff(fit.cdf(edges))
    else:  # EHquad: proportional heights at the midpoints
        heights = fit.density(mids) * np.diff(edges)
        weights = heights / heights.sum()
    return Quadrature(mids, weights, method, npts)
