"""Tests for BBClassify.

The reference values are the output printed in the BB-CLASS manual
(Brennan, 2004, Tables 2-4 and 6-8), which BBClassify must reproduce when it
is given the same inputs. For the LL example that means supplying the
observed minimum and maximum (101 and 190) as the possible scores, because
that is what BB-CLASS used.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import bbclassify
from bbclassify import BBClassifyError
from bbclassify.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
TOL = 6e-6  # the manual prints five decimals


@pytest.fixture
def workdir(tmp_path):
    for name in ("act288m", "LL data"):
        shutil.copy(EXAMPLES / name, tmp_path / name)
    return tmp_path


def run(folder: Path, cards: str, name: str = "cc"):
    path = folder / name
    path.write_text(cards)
    return bbclassify.run(path)


def tables(r):
    return [t.matrix for t in r.tables]


HB_CARDS = 'HB 0. 4 check\n"act288m" f 1 2\n2   24.\n'
LL_CARDS = 'LL 0.9 4    check\n"LL data" f 1 2 190 101\n3 140. 160.    .4      .6\n'


def test_hanson_brennan_manual_example(workdir):
    r = run(workdir, HB_CARDS)
    assert r.n == 40 and r.k == 0 and r.n_examinees == 151050
    np.testing.assert_allclose(r.observed_proportions, [.80351, .19649], atol=TOL)
    f = r.fit
    np.testing.assert_allclose([f.alpha, f.beta, f.lower, f.upper], [.523779, 1.625693, .223172, 1.0], atol=1e-6)
    assert f.moments_fit == 3
    m = r.model_moments
    np.testing.assert_allclose([m.mean, m.sd, m.skew, m.kurt], [16.498709, 8.048720, .829364, 2.965899], atol=1e-6)
    m = r.fitted_moments
    np.testing.assert_allclose([m.mean, m.sd, m.skew, m.kurt], [16.498709, 8.048720, .829364, 2.925241], atol=1e-6)
    m = r.true_moments
    np.testing.assert_allclose([m.mean, m.sd, m.skew, m.kurt], [16.498709, 7.516707, 1.021434, 3.158296], atol=1e-6)
    assert r.chi_square.likelihood_ratio == pytest.approx(339.84519, abs=1e-4)
    assert r.chi_square.pearson == pytest.approx(344.66484, abs=1e-4)
    assert r.reliability_from_moments == pytest.approx(.87217, abs=TOL)
    assert r.sem_from_moments == pytest.approx(2.87767, abs=TOL)
    assert r.bivariate_sum == pytest.approx(1.0, abs=TOL)
    expected = [
        [[.78247, .03795], [.01778, .16180]],
        [[.76127, .03898], [.03898, .16076]],
        [[.78565, .03733], [.01785, .15916]],
        [[.76437, .03835], [.03914, .15814]],
    ]
    for got, want in zip(tables(r), expected):
        np.testing.assert_allclose(got, want, atol=TOL)
    acc, con, acc_actual, con_actual = r.tables
    assert acc.correct == pytest.approx(.94427, abs=TOL)
    assert acc.false_positive == pytest.approx(.03795, abs=TOL)
    assert acc.false_negative == pytest.approx(.01778, abs=TOL)
    assert (con.correct, con.chance, con.kappa) == pytest.approx((.92204, .68031, .75613), abs=TOL)
    assert (con_actual.correct, con_actual.chance, con_actual.kappa) == pytest.approx((.92251, .68375, .75498), abs=TOL)


def test_livingston_lewis_manual_example(workdir):
    r = run(workdir, LL_CARDS)
    assert r.effective_length == pytest.approx(49.70252, abs=TOL)
    assert r.n == 50
    np.testing.assert_allclose(r.model_cuts, [21.91011, 33.14607], atol=TOL)
    np.testing.assert_allclose(r.observed_proportions, [.214, .313, .473], atol=TOL)
    f = r.fit
    np.testing.assert_allclose([f.alpha, f.beta, f.lower, f.upper], [2.666934, 1.302899, 0, .907239], atol=1e-6)
    m = r.model_moments
    np.testing.assert_allclose([m.mean, m.sd], [30.474157, 10.067981], atol=1e-6)
    m = r.true_moments
    np.testing.assert_allclose([m.mean, m.sd, m.skew, m.kurt], [30.474157, 9.554546, -.546516, 2.561804], atol=1e-6)
    assert r.chi_square.likelihood_ratio == pytest.approx(158.33372, abs=1e-4)
    assert r.chi_square.pearson == pytest.approx(151.27795, abs=1e-4)
    assert r.reliability_from_moments == pytest.approx(.90061, abs=TOL)
    expected = [
        [[.14951, .01090, 0], [.06173, .20016, .01146], [.00045, .12324, .44255]],
        [[.16625, .04479, .00066], [.04479, .22121, .06830], [.00066, .06830, .38505]],
        [[.15114, .01021, 0], [.06240, .18741, .01194], [.00046, .11539, .46106]],
        [[.16806, .04193, .00068], [.04527, .20712, .07116], [.00066, .06395, .40116]],
    ]
    for got, want in zip(tables(r), expected):
        np.testing.assert_allclose(got, want, atol=TOL)
    assert r.tables[0].false_positive == pytest.approx(.02236, abs=TOL)
    assert r.tables[0].false_negative == pytest.approx(.18542, abs=TOL)


def test_moments_input_matches_manual(workdir):
    (workdir / "HBdata moments").write_text("151050 16.498709 8.048720 .829364 2.965899 0 40 .80351 .19649\n")
    hb = run(workdir, 'HB 0. 4 check\n"HBdata moments" m\n2   24.\n')
    full = run(workdir, HB_CARDS, "full")
    for a, b in zip(tables(hb), tables(full)):
        np.testing.assert_allclose(a, b, atol=2e-6)
    assert hb.chi_square is None

    (workdir / "LL data moments").write_text("1000\n155.244000\n17.921006\n-.499583\n2.543831\n101\n190\n.214\n.313\n.473\n")
    ll = run(workdir, 'LL 0.9 4    check\n"LL data moments" m\n3 140. 160.    .4 .6\n')
    full = run(workdir, LL_CARDS, "fullLL")
    for a, b in zip(tables(ll), tables(full)):
        np.testing.assert_allclose(a, b, atol=2e-6)
    assert any("POSSIBLE" in note for note in ll.feedback.notes)


def test_raw_scores_match_frequencies(workdir):
    freq = np.loadtxt(workdir / "act288m")
    raw = np.repeat(freq[:, 0], freq[:, 1].astype(int)).astype(int)
    np.savetxt(workdir / "raw", raw, fmt="%d")
    r_raw = run(workdir, 'HB 0. 4\n"raw" r 1 40\n2 24\n', "raw_cc")
    r_freq = run(workdir, HB_CARDS)
    for a, b in zip(tables(r_raw), tables(r_freq)):
        np.testing.assert_allclose(a, b, atol=1e-12)

    ll = np.loadtxt(workdir / "LL data")
    np.savetxt(workdir / "llraw", np.repeat(ll[:, 0], ll[:, 1].astype(int)), fmt="%d")
    r_raw = run(workdir, 'LL 0.9 4\n"llraw" r 1 190 101\n3 140 160 .4 .6\n', "llraw_cc")
    for a, b in zip(tables(r_raw), tables(run(workdir, LL_CARDS))):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_possible_range_changes_effective_test_length(workdir):
    """The point of BBClassify: the possible, not observed, range enters the effective test length."""
    observed = run(workdir, LL_CARDS)
    possible = run(workdir, 'LL 0.9 4\n"LL data" f 1 2 200 0\n3 140 160 .4 .6\n', "possible")
    assert possible.effective_length != pytest.approx(observed.effective_length)
    mean, var, rel = 155.244, 17.921006 ** 2, .9
    assert possible.effective_length == pytest.approx((mean * (200 - mean) - rel * var) / (var * (1 - rel)), rel=1e-6)
    assert any("observed range" in note for note in possible.feedback.notes)
    default_min = run(workdir, 'LL 0.9 4\n"LL data" f 1 2 200\n3 140 160\n', "default_min")
    assert default_min.xmin == 0
    assert any("xmin" in note for note in default_min.feedback.notes)


def test_compound_binomial_and_options(workdir):
    r = run(workdir, 'HB 0.9 4 xfit 5 check\n"act288m" f 1 2\n3 20 30\n')
    assert r.k > 0
    for t in r.tables:
        assert t.matrix.sum() == pytest.approx(1, abs=1e-6)
    assert r.chi_square.threshold == 5
    two = run(workdir, 'HB 0. 2\n"act288m" f 1 2\n2 24\n', "two")
    assert (two.fit.lower, two.fit.upper, two.fit.moments_fit) == (0, 1, 2)
    ed = run(workdir, HB_CARDS, "ed")
    eh = run(workdir, 'HB 0. 4 EHquad npts 2000\n"act288m" f 1 2\n2 24\n', "eh")
    gq = run(workdir, 'HB 0. 4 Gquad\n"act288m" f 1 2\n2 24\n', "gq")
    # alpha < 1 here, so the density is infinite at the lower limit: EHquad and Gquad are
    # only roughly right, and both warn.
    np.testing.assert_allclose(tables(eh)[1], tables(ed)[1], atol=3e-3)
    np.testing.assert_allclose(tables(gq)[1], tables(ed)[1], atol=3e-2)
    assert any("EHquad is inaccurate" in w for w in eh.feedback.warnings)
    assert any("Gauss quadrature" in w for w in gq.feedback.warnings)
    assert not ed.feedback.warnings
    smooth = run(workdir, 'LL 0.9 4 EHquad\n"LL data" f 1 2 190 101\n3 140 160 .4 .6\n', "smooth")
    np.testing.assert_allclose(tables(smooth)[1], tables(run(workdir, LL_CARDS, "ll"))[1], atol=1e-4)


@pytest.mark.parametrize("cards, message", [
    ('XX 0. 4\n"act288m" f 1 2\n2 24\n', "procedure must be HB"),
    ('HB 1.2 4\n"act288m" f 1 2\n2 24\n', "reliability must be at least 0"),
    ('LL 0 4\n"LL data" f 1 2 190\n2 150\n', "greater than 0"),
    ('HB 0. 3\n"act288m" f 1 2\n2 24\n', "must be 2 or 4"),
    ('HB 0. 4 quick\n"act288m" f 1 2\n2 24\n', "not a recognized option"),
    ('HB 0. 4\n"missing file" f 1 2\n2 24\n', "was not found"),
    ('LL 0.9 4\n"LL data" f 1 2\n3 140 160\n', "maximum POSSIBLE score"),
    ('LL 0.9 4\n"LL data" f 1 2 180\n3 140 160\n', "outside the possible range"),
    ('HB 0. 4\n"act288m" f 1 2\n3 24\n', "with K = 3 categories"),
    ('HB 0. 4\n"act288m" f 1 2\n2 45\n', "cut scores must be"),
    ('HB 0. 4\n"act288m" f 1 2\n2 24 1.5\n', "between 0 and 1"),
    ('HB 0. 4\n"act288m" f 3 2\n2 24\n', r"column\(s\)"),
    ('HB 0.5 4\n"act288m" f 1 2\n2 24\n', None),
    ('HB 0. 4\n"act288m" f 1 2\n', "three control cards"),
])
def test_errors_are_explained(workdir, cards, message):
    if message is None:  # runs, but warns that Lord's k is negative
        r = run(workdir, cards)
        assert any("lower than KR-21" in w for w in r.feedback.warnings)
        return
    with pytest.raises(BBClassifyError, match=message):
        run(workdir, cards)


def test_hb_frequency_file_must_list_every_score(workdir):
    lines = (workdir / "act288m").read_text().splitlines()[1:]  # drop the score-0 line
    (workdir / "short").write_text("\n".join(lines) + "\n")
    with pytest.raises(BBClassifyError, match="one line for every possible score"):
        run(workdir, 'HB 0. 4\n"short" f 1 2\n2 24\n')


def test_non_numeric_data_is_reported_with_line(workdir):
    (workdir / "bad").write_text("score\n12\n13\n")
    with pytest.raises(BBClassifyError, match="line 1: 'score'"):
        run(workdir, 'HB 0. 4\n"bad" r 1 40\n2 24\n')


def test_console_program(workdir, capsys):
    (workdir / "cc").write_text(HB_CARDS)
    assert main([str(workdir / "cc")]) == 0
    out = capsys.readouterr().out
    assert "Successful execution" in out
    for suffix in ("out", "true dist", "observed dist"):
        assert (workdir / f"cc {suffix}.txt").is_file()
    (workdir / "bad").write_text('HB 0. 4\n"act288m" f 1 2\n2 45\n')
    assert main([str(workdir / "bad")]) == 1
    assert "ERROR" in capsys.readouterr().out
