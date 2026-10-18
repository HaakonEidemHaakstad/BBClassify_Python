"""The NumPy-only special functions agree with SciPy (a test-only dependency)."""

import numpy as np
import pytest

from bbclassify import special

scipy_special = pytest.importorskip("scipy.special")
scipy_stats = pytest.importorskip("scipy.stats")

SHAPES = [(0.3, 0.4), (0.52, 1.63), (1, 1), (2.67, 1.30), (5, 3), (40, 0.8), (150, 220)]


@pytest.mark.parametrize("a, b", SHAPES)
def test_beta_cdf_and_pdf(a, b):
    x = np.concatenate([[0, 1e-12, 1e-6], np.linspace(0, 1, 1001), [1 - 1e-6, 1]])
    np.testing.assert_allclose(special.beta_cdf(x, a, b), scipy_stats.beta.cdf(x, a, b), rtol=1e-11, atol=1e-14)
    inner = (x > 0) & (x < 1)
    np.testing.assert_allclose(special.beta_pdf(x[inner], a, b), scipy_stats.beta.pdf(x[inner], a, b), rtol=1e-10)
    assert special.beta_pdf(np.array([-0.1, 1.1]), a, b).tolist() == [0.0, 0.0]


@pytest.mark.parametrize("n", [1, 2, 10, 40, 300])
def test_binomial(n):
    p = np.concatenate([[0, 1e-9, 1], np.linspace(0, 1, 101)])
    expected = scipy_stats.binom.pmf(np.arange(n + 1)[None, :], n, p[:, None])
    np.testing.assert_allclose(special.binom_pmf_matrix(n, p), expected, rtol=1e-10, atol=1e-300)


@pytest.mark.parametrize("m, alpha, beta", [(5, 0.0, 0.0), (43, 0.63, -0.48), (53, 0.3, 1.67), (303, 10.0, -0.5)])
def test_gauss_jacobi(m, alpha, beta):
    x, w = special.gauss_jacobi(m, alpha, beta)
    xs, ws = scipy_special.roots_jacobi(m, alpha, beta)
    np.testing.assert_allclose(np.sort(x), xs, atol=1e-12)
    # Weights far below machine precision relative to the largest weight are not
    # resolved by the eigenvector method, so compare on an absolute scale and,
    # more to the point, compare integrals of polynomials up to degree 2m - 1.
    np.testing.assert_allclose(w[np.argsort(x)], ws / ws.sum(), rtol=1e-6, atol=1e-12)
    y, ys = (x + 1) / 2, (xs + 1) / 2
    for degree in (1, m // 2, 2 * m - 1):
        assert w @ y ** degree == pytest.approx((ws / ws.sum()) @ ys ** degree, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("df", [1, 2, 5, 37, 47, 300])
def test_chi2_sf(df):
    for x in [0, 0.01, 1, df / 2, df, 2 * df, 5 * df + 50, 340.6]:
        assert special.chi2_sf(x, df) == pytest.approx(scipy_stats.chi2.sf(x, df), rel=1e-10, abs=1e-300)


def test_find_root():
    assert special.find_root(lambda x: x ** 3 - 2, 0, 2) == pytest.approx(2 ** (1 / 3), abs=1e-14)
