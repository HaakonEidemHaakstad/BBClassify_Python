"""Hanson and Brennan (HB) and Livingston and Lewis (LL) classification analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .controlcards import ControlCards
from .data import ScoreData
from .feedback import BBClassifyError, Feedback
from .special import chi2_sf
from .model import (BetaFit, Moments, Quadrature, central_from_raw, distribution_moments,
                    effective_test_length, error_pmf, fit_beta, lords_k, marginal_pmf, quadrature,
                    true_score_moments)


@dataclass
class Table:
    """A K x K contingency table with its summary statistics."""

    title: str
    matrix: np.ndarray
    row_prefix: str
    col_prefix: str
    kind: str  # "accuracy" or "consistency"

    @property
    def row_marg(self) -> np.ndarray:
        return self.matrix.sum(axis=1)

    @property
    def col_marg(self) -> np.ndarray:
        return self.matrix.sum(axis=0)

    @property
    def correct(self) -> float:
        return float(np.trace(self.matrix))

    @property
    def false_positive(self) -> float:
        return float(np.triu(self.matrix, 1).sum())

    @property
    def false_negative(self) -> float:
        return float(np.tril(self.matrix, -1).sum())

    @property
    def chance(self) -> float:
        return float(np.sum(self.row_marg * self.col_marg))

    @property
    def kappa(self) -> float:
        c = self.chance
        return (self.correct - c) / (1 - c) if abs(1 - c) > 1e-12 else float("nan")


@dataclass
class ChiSquare:
    likelihood_ratio: float
    pearson: float
    df: int
    cells: int
    threshold: float

    @property
    def p_likelihood_ratio(self) -> float:
        return chi2_sf(self.likelihood_ratio, self.df) if self.df > 0 else float("nan")

    @property
    def p_pearson(self) -> float:
        return chi2_sf(self.pearson, self.df) if self.df > 0 else float("nan")


@dataclass
class Results:
    cards: ControlCards
    data: ScoreData
    feedback: Feedback
    n_examinees: float
    xmin: float
    xmax: float
    n: int                         # test length (HB) or rounded effective test length (LL)
    effective_length: float | None  # unrounded effective test length (LL)
    k: float
    xcuts: np.ndarray
    model_cuts: np.ndarray         # cuts on the 0..n model score scale
    tcuts: np.ndarray
    observed_proportions: np.ndarray
    raw_moments: Moments           # moments of the reported scores
    model_moments: Moments         # moments of the 0..n model scores (= raw for HB)
    fit: BetaFit
    fitted_moments: Moments
    true_moments: Moments          # on the 0..n model scale
    reliability_from_moments: float
    sem_from_moments: float
    quad: Quadrature
    bivariate_sum: float | None
    chi_square: ChiSquare | None
    tables: list[Table]
    fitted_pmf: np.ndarray
    observed_counts: np.ndarray | None
    true_distribution: np.ndarray = field(repr=False)


def _category_membership(n: int, cuts: np.ndarray, procedure: str) -> np.ndarray:
    """Matrix (n + 1, K) giving the share of each model score in each observed category.

    HB scores are counts, so score i is in the first category whose lower cut
    is at or below i. LL scores are rescaled reported scores rounded to the
    0..n scale, so score i stands for the interval [i - .5, i + .5] and is
    shared between categories in proportion to how much of that interval
    lies on each side of a cut (the linear interpolation BB-CLASS uses).
    """
    scores = np.arange(n + 1)
    K = len(cuts) + 1
    if procedure == "HB":
        category = np.searchsorted(cuts - 1e-9, scores, side="right")
        member = np.zeros((n + 1, K))
        member[scores, category] = 1.0
        return member
    bounds = np.concatenate([[-np.inf], cuts, [np.inf]])
    lo, hi = scores - 0.5, scores + 0.5
    return np.stack([np.clip(np.minimum(hi, bounds[j + 1]) - np.maximum(lo, bounds[j]), 0, 1)
                     for j in range(K)], axis=1)


def analyze(cards: ControlCards, data: ScoreData) -> Results:
    """Run the analysis specified by the control cards on the data."""
    fb = Feedback()
    fb.notes.extend(cards.notes)
    fb.warnings.extend(cards.warnings)
    fb.notes.extend(data.notes)
    fb.warnings.extend(data.warnings)
    procedure = cards.procedure

    # -- Score scale ---------------------------------------------------------
    if procedure == "HB":
        n = data.nitems
        xmin, xmax = 0.0, float(n)
    else:
        xmin, xmax = data.xmin, data.xmax
    if not xmax > xmin:
        raise BBClassifyError(f"The maximum possible score ({xmax:g}) must be greater than the minimum possible score ({xmin:g}).")

    # -- Observed scores -----------------------------------------------------
    if data.moments is not None:
        raw = data.moments
        n_examinees = data.n_examinees
        observed_props = data.category_proportions
    else:
        raw = distribution_moments(data.scores, data.freqs)
        n_examinees = float(data.freqs.sum())
        observed_cat = np.searchsorted(np.asarray(cards.xcuts) - 1e-9, data.scores, side="right")
        observed_props = np.bincount(observed_cat, weights=data.freqs, minlength=cards.ncat) / n_examinees
        if np.min(data.scores) < xmin - 1e-9 or np.max(data.scores) > xmax + 1e-9:
            where = "the test length (number of items)" if procedure == "HB" else "the possible minimum and maximum scores"
            raise BBClassifyError(
                f"Observed scores range from {np.min(data.scores):g} to {np.max(data.scores):g}, which is outside "
                f"the possible range {xmin:g} to {xmax:g}. Check {where} on control card 2.")
        if procedure == "LL":
            obs_lo, obs_hi = float(np.min(data.scores)), float(np.max(data.scores))
            if obs_lo > xmin or obs_hi < xmax:
                fb.note(
                    f"Observed scores range from {obs_lo:g} to {obs_hi:g}; the effective test length uses the possible range "
                    f"{xmin:g} to {xmax:g} as specified on control card 2. (BB-CLASS would have used the observed range, "
                    "which is the error this program corrects.)")
    if procedure == "LL" and data.moments is not None:
        fb.note(
            f"The minimum ({xmin:g}) and maximum ({xmax:g}) in the moments file are used as the lowest and highest POSSIBLE "
            "scores on the test. If they are the lowest and highest OBSERVED scores (as in the BB-CLASS manual's example), "
            "replace them with the possible scores; otherwise the effective test length is biased.")

    xcuts = np.asarray(cards.xcuts, dtype=float)
    if np.any(xcuts <= xmin) or np.any(xcuts > xmax):
        raise BBClassifyError(
            f"Observed cut scores must be greater than the minimum possible score ({xmin:g}) and not greater than the "
            f"maximum possible score ({xmax:g}); control card 3 gives {', '.join(f'{c:g}' for c in xcuts)}.")
    if cards.tcuts is None:
        tcuts = (xcuts - xmin) / (xmax - xmin)
        fb.note("True cut scores were not given on control card 3, so they were set to the proportion-correct "
                "observed cut scores, (xcut - Xmin) / (Xmax - Xmin).")
    else:
        tcuts = np.asarray(cards.tcuts, dtype=float)

    # -- Error model and model score scale ----------------------------------
    rel = cards.reliability
    effective = None
    if procedure == "HB":
        k = 0.0 if rel == 0 else lords_k(raw.mean, raw.var, rel, n)
        if k < 0:
            fb.warn(f"Lord's k is negative ({k:.5f}), which the compound binomial model does not allow for. This happens when "
                    f"the reliability is lower than KR-21 ({_kr21(raw.mean, raw.var, n):.5f} for these scores), the value for "
                    "equally difficult items. "
                    "Check the reliability, or use 0 for the binomial error model.")
        model_moments = raw
        model_cuts = xcuts
    else:
        k = 0.0
        effective = effective_test_length(raw.mean, raw.var, rel, xmin, xmax)
        if not effective >= 1.5:
            raise BBClassifyError(
                f"The effective test length is {effective:.5f}, which is too short for the analysis (it must round to at least 2). "
                "Check the reliability on control card 1 and the possible minimum and maximum scores on control card 2.")
        n = int(math.floor(effective + 0.5))
        scale = n / (xmax - xmin)
        model_moments = raw.rescaled(scale, xmin)
        model_cuts = (xcuts - xmin) * scale

    # -- True-score distribution ---------------------------------------------
    estimated_true = central_from_raw(true_score_moments(model_moments, n, k))
    fit = fit_beta(estimated_true, cards.nparm, fb)
    true_moments = estimated_true.rescaled(n)

    quad = quadrature(fit, cards.quadrature, cards.npts, tcuts)
    if cards.quadrature != "EDquad" and (fit.alpha < 1 or fit.beta < 1):
        name = "Gauss quadrature (Gquad)" if cards.quadrature == "Gquad" else "EHquad"
        fb.warn(f"{name} is inaccurate when a beta shape parameter is below 1, because the true-score density is then "
                f"infinite at a limit (alpha = {fit.alpha:.4f}, beta = {fit.beta:.4f}). EDquad is recommended.")
    pmf_given_tau = error_pmf(n, k, quad.nodes)              # nodes x scores
    fitted_pmf = marginal_pmf(fit, n, k)
    if k != 0 and np.any(fitted_pmf < -1e-10):
        fb.warn("Lord's two-term approximation gives negative probabilities for some observed scores. "
                "Results may be unreliable; consider a lower reliability or the binomial error model (reliability 0).")

    member = _category_membership(n, model_cuts, procedure)
    cat_given_tau = pmf_given_tau @ member                   # nodes x categories
    K = cards.ncat
    true_cat = np.searchsorted(tcuts, quad.nodes, side="right")
    weighted = cat_given_tau * quad.weights[:, None]

    accuracy = np.zeros((K, K))
    for t in range(K):
        accuracy[t] = weighted[true_cat == t].sum(axis=0)
    consistency = weighted.T @ cat_given_tau

    def rescale_columns(matrix: np.ndarray, label: str) -> np.ndarray:
        expected = matrix.sum(axis=0)
        out = np.zeros_like(matrix)
        for j in range(K):
            if expected[j] > 0:
                out[:, j] = matrix[:, j] * observed_props[j] / expected[j]
            elif observed_props[j] > 0:
                fb.warn(f"The model predicts no examinees in observed category x{j} although "
                        f"{observed_props[j]:.5f} of the data are in it; the {label} table cannot account for them.")
        return out

    tables = [
        Table("ACCURACY RELATIVE TO EXPECTED OBSERVED SCORES GIVEN MODEL", accuracy, "t", "x", "accuracy"),
        Table("CONSISTENCY USING EXPECTED OBSERVED SCORES GIVEN MODEL", consistency, "x", "x", "consistency"),
        Table("ACCURACY RELATIVE TO ACTUAL OBSERVED SCORES",
              rescale_columns(accuracy, "actual-observed accuracy"), "t", "x", "accuracy"),
        Table("CONSISTENCY USING EXPECTED (row) VS. ACTUAL (column) OBSERVED SCORES",
              rescale_columns(consistency, "actual-observed consistency"), "x", "x", "consistency"),
    ]
    for j, p in enumerate(observed_props):
        if p == 0:
            fb.warn(f"No examinees in the data are in observed category x{j}. Check the cut scores on control card 3.")

    bivariate = None
    if cards.check:
        bivariate = float(np.sum(quad.weights * pmf_given_tau.sum(axis=1) ** 2))
        if abs(bivariate - 1) > 1e-4:
            fb.warn(f"The bivariate probabilities sum to {bivariate:.5f} rather than 1, so the numerical integration "
                    "may be inaccurate. Try EDquad, or more quadrature points (npts).")

    # -- Fit, reliability ----------------------------------------------------
    scores = np.arange(n + 1)
    fitted_moments = distribution_moments(scores, fitted_pmf)
    reliability = true_moments.var / model_moments.var
    sem = model_moments.sd * math.sqrt(max(1 - reliability, 0.0))

    chi = None
    observed_counts = None
    if data.moments is None:
        model_scores = data.scores if procedure == "HB" else np.floor((data.scores - xmin) * n / (xmax - xmin) + 0.5)
        observed_counts = np.bincount(model_scores.astype(int), weights=data.freqs, minlength=n + 1)
        expected_counts = fitted_pmf * n_examinees
        cells = expected_counts > cards.xfit
        if cells.sum() > 0:
            o, e = observed_counts[cells], expected_counts[cells]
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = 2 * float(np.sum(np.where(o > 0, o * np.log(o / e), 0.0)))
            pearson = float(np.sum((o - e) ** 2 / e))
            df = int(cells.sum()) - 1 - fit.moments_fit
            chi = ChiSquare(lr, pearson, df, int(cells.sum()), cards.xfit)
            if df <= 0:
                fb.warn(f"Only {cells.sum()} score cells have fitted frequencies above {cards.xfit:g}, which leaves no degrees "
                        "of freedom for the chi-square tests. Lower xfit on control card 1.")

    true_distribution = _true_distribution(fit, cards.npts)

    if n_examinees < 100 and data.moments is None:
        fb.warn(f"Only {n_examinees:g} examinees: method-of-moments estimates of the true-score distribution are "
                "unstable in small samples, so interpret the results with caution.")

    return Results(cards=cards, data=data, feedback=fb, n_examinees=n_examinees, xmin=xmin, xmax=xmax,
                   n=n, effective_length=effective, k=k, xcuts=xcuts, model_cuts=model_cuts, tcuts=tcuts,
                   observed_proportions=observed_props, raw_moments=raw, model_moments=model_moments, fit=fit,
                   fitted_moments=fitted_moments, true_moments=true_moments,
                   reliability_from_moments=reliability, sem_from_moments=sem, quad=quad,
                   bivariate_sum=bivariate, chi_square=chi, tables=tables, fitted_pmf=fitted_pmf,
                   observed_counts=observed_counts, true_distribution=true_distribution)


def _kr21(mean: float, var: float, n: int) -> float:
    return n / (n - 1) * (1 - mean * (n - mean) / (n * var))


def _true_distribution(fit: BetaFit, npts: int) -> np.ndarray:
    """Midpoint, height, interval probability and upper-limit cdf for npts intervals on [l, u]."""
    edges = np.linspace(fit.lower, fit.upper, npts + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    cdf = fit.cdf(edges)
    return np.column_stack([mids, fit.density(mids), np.diff(cdf), cdf[1:]])
