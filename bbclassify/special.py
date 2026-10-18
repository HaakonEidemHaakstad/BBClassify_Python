"""Special functions, implemented with NumPy only.

SciPy is avoided because importing it dominated the program's start-up time
(and its size when packaged), while only these few functions are needed. The
tests compare each of them against SciPy.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-15
_TINY = 1e-300


def _xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x * log(y), with 0 * log(0) = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x == 0, 0.0, x * np.log(y))


def _xlog1py(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x * log(1 + y), with 0 * log(0) = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x == 0, 0.0, x * np.log1p(y))


def binom_pmf_matrix(n: int, p: np.ndarray) -> np.ndarray:
    """Binomial probabilities P(X = i | p), i = 0..n, as a (len(p), n + 1) matrix."""
    p = np.asarray(p, dtype=float)[:, None]
    i = np.arange(n + 1, dtype=float)[None, :]
    lchoose = np.array([math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) for j in range(n + 1)])[None, :]
    return np.exp(lchoose + _xlogy(i, p) + _xlog1py(n - i, -p))


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Density of the standard beta distribution (zero outside [0, 1])."""
    x = np.asarray(x, dtype=float)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    inside = (x >= 0) & (x <= 1)
    xi = np.where(inside, x, 0.5)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.exp(log_norm + _xlogy(a - 1, xi) + _xlog1py(b - 1, -xi))
    return np.where(inside, out, 0.0)


def _beta_continued_fraction(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Continued fraction for the incomplete beta function (modified Lentz), vectorized."""
    qab, qap, qam = a + b, a + 1, a - 1
    c = np.ones_like(x)
    d = 1 - qab * x / qap
    d = np.where(np.abs(d) < _TINY, _TINY, d)
    d = 1 / d
    h = d.copy()
    done = np.zeros(x.shape, dtype=bool)
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = 1 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1 / d
        h = np.where(done, h, h * d * c)
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = 1 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1 / d
        delta = d * c
        h = np.where(done, h, h * delta)
        done |= np.abs(delta - 1) < _EPS
        if done.all():
            break
    return h


def beta_cdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Regularized incomplete beta function I_x(a, b)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    out = np.where(x >= 1, 1.0, 0.0)
    inner = (x > 0) & (x < 1)
    if not inner.any():
        return out
    xi = x[inner]
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * np.log(xi) + b * np.log1p(-xi))
    front = np.exp(log_front)
    direct = xi < (a + 1) / (a + b + 2)
    # Use the symmetry I_x(a, b) = 1 - I_{1-x}(b, a) where the fraction converges faster.
    aa = np.where(direct, a, b)
    bb = np.where(direct, b, a)
    xx = np.where(direct, xi, 1 - xi)
    cf = _beta_continued_fraction(aa, bb, xx)
    out[inner] = np.where(direct, front * cf / a, 1 - front * cf / b)
    return out


def gauss_jacobi(m: int, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and normalized weights of m-point Gauss-Jacobi quadrature on [-1, 1].

    Weight function (1 - x)^alpha (1 + x)^beta (Golub-Welsch algorithm). The
    weights are scaled to sum to 1.
    """
    k = np.arange(m, dtype=float)
    s = 2 * k + alpha + beta
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = (beta ** 2 - alpha ** 2) / (s * (s + 2))
    diag[0] = (beta - alpha) / (alpha + beta + 2)
    j = k[1:]
    s1 = 2 * j + alpha + beta
    off = np.sqrt(4 * j * (j + alpha) * (j + beta) * (j + alpha + beta) / (s1 ** 2 * (s1 + 1) * (s1 - 1)))
    matrix = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    nodes, vectors = np.linalg.eigh(matrix)
    weights = vectors[0] ** 2
    return nodes, weights / weights.sum()


def find_root(f, lo: float, hi: float, tol: float = 1e-15) -> float:
    """Root of f in [lo, hi], where f(lo) and f(hi) have opposite signs (bisection)."""
    f_lo = f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if f_mid == 0 or (hi - lo) / 2 < tol:
            return mid
        if (f_mid < 0) == (f_lo < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def _upper_gamma_regularized(a: float, x: float) -> float:
    """Q(a, x) = Gamma(a, x) / Gamma(a), by series or continued fraction."""
    if x <= 0:
        return 1.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = total = 1 / a
        ap = a
        for _ in range(10000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        return 1 - total * math.exp(log_front)
    b = x + 1 - a
    c = 1 / _TINY
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = _TINY if abs(d) < _TINY else d
        c = b + an / c
        c = _TINY if abs(c) < _TINY else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < _EPS:
            break
    return math.exp(log_front) * h


def chi2_sf(x: float, df: int) -> float:
    """Upper-tail probability of the chi-square distribution."""
    if df <= 0:
        return float("nan")
    return _upper_gamma_regularized(df / 2, x / 2)
