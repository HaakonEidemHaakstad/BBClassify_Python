"""Output files, in the layout of BB-CLASS."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import __version__
from .analysis import Results, Table

RULE = " " + "*" * 72


def _header() -> list[str]:
    title = [
        "BBClassify: Beta-Binomial Classification Consistency and Accuracy",
        f"Version {__version__}",
        "",
        "Haakon Eidem Haakstad",
        "",
        "A corrected implementation of the procedures in BB-CLASS",
        "(Brennan, 2004). MIT License.",
    ]
    lines = [RULE]
    for t in title:
        lines.append(" *** " + t.center(64) + " ***")
    lines.append(RULE)
    return lines


def _table(table: Table) -> list[str]:
    K = table.matrix.shape[0]
    head = "      " + "".join(f"{table.col_prefix}{j:<1}".rjust(10) for j in range(K)) + "      marg"
    lines = [f" ***{table.title}***", "", head]
    for i in range(K):
        row = table.matrix[i]
        lines.append(f"{table.row_prefix + str(i):>5} " + "".join(f"{v:10.5f}" for v in row) + f"{row.sum():10.5f}")
    lines.append("")
    lines.append(" marg " + "".join(f"{v:10.5f}" for v in table.col_marg) + f"{table.matrix.sum():10.5f}")
    lines.append("")
    if table.kind == "accuracy":
        lines.append(f" probability of correct classification = {table.correct:.5f}")
        lines.append(f" false positive rate = {table.false_positive:.5f}; false negative rate = {table.false_negative:.5f}")
    else:
        lines.append(f" pc = {table.correct:.5f}; pchance = {table.chance:.5f}; kappa = {table.kappa:.5f}")
        lines.append(f" probability of misclassification = {1 - table.correct:.5f}")
    lines.append("")
    return lines


def _moment_row(label: str, m, extra: str = "") -> str:
    return f"{label:>11} {m.mean:12.6f} {m.sd:12.6f} {m.skew:12.6f} {m.kurt:11.6f}{extra}"


def main_output(r: Results) -> str:
    c = r.cards
    proc = "Hanson and Brennan" if c.procedure == "HB" else "Livingston and Lewis"
    out = _header()
    out += ["", f" *** {proc} Results ***", "", f" *** Listing of Control Cards in {c.path.name} ***", ""]
    out += [" " + line for line in c.lines]
    out += ["", " " + "*" * 47, ""]

    out.append(f"              Number of examinees = {r.n_examinees:15.5f}")
    out.append(f"                Input reliability = {c.reliability:15.5f}")
    if c.procedure == "HB":
        out.append(f"                      Test length = {r.n:9d}")
    else:
        out.append(f"   Minimum possible score (Xmin) = {r.xmin:15.5f}")
        out.append(f"   Maximum possible score (Xmax) = {r.xmax:15.5f}")
        out.append(f"            Effective test length = {r.effective_length:15.5f}")
        out.append(f"  Effective test length (rounded) = {r.n:9d}")
    model = "(binomial error model)" if r.k == 0 else "(compound binomial error model)"
    out.append(f"                         Lord's k = {r.k:15.5f}   {model}")
    out += ["", f" Number of Categories = {c.ncat}", ""]
    out.append(" Cut Scores:    xcut[] = " + "".join(f"{v:11.5f}" for v in r.xcuts))
    if c.procedure == "LL":
        out.append("           xprimecut[] = " + "".join(f"{v:11.5f}" for v in r.model_cuts))
    out.append("                tcut[] = " + "".join(f"{v:11.5f}" for v in r.tcuts))
    out += ["", " Category proportions in original data:", "           " + "".join(f"{v:10.5f}" for v in r.observed_proportions), ""]

    if c.procedure == "LL":
        span = r.xmax - r.xmin
        p = r.raw_moments.rescaled(1 / span, r.xmin)
        out += [" ***Moments used by Livingston and Lewis Procedure***", "",
                "                    Mean         S.D.        Skew        Kurt          Min          Max", ""]
        out.append(_moment_row("x", r.raw_moments, f" {r.xmin:12.6f} {r.xmax:12.6f}"))
        out.append(_moment_row("p", p, f" {0:12.6f} {1:12.6f}"))
        out.append(_moment_row("Raw=x'", r.model_moments, f" {0:12.6f} {r.n:12.6f}"))
        out += ["", " (Xmin and Xmax are the lowest and highest possible scores, not the observed ones.)", ""]

    f = r.fit
    out += [" ***Parameter Estimates for Beta Distribution***", "",
            "        alpha         beta    low limit    upp limit", "",
            f" {f.alpha:12.6f} {f.beta:12.6f} {f.lower:12.6f} {f.upper:12.6f}", "",
            f" Number of moments fit: {f.moments_fit:8d}", "",
            " ***Moments (Raw, Fitted Raw, True)***", "",
            "                    Mean         S.D.        Skew       Kurt", ""]
    out.append(_moment_row("Raw", r.model_moments))
    out.append(_moment_row("Fitted Raw", r.fitted_moments))
    out.append(_moment_row("True", r.true_moments))
    out.append("")
    if r.chi_square is not None:
        chi = r.chi_square
        out.append(f" Likelihood Ratio Chi-Square = {chi.likelihood_ratio:.5f} (with df = {chi.df}; p = {chi.p_likelihood_ratio:.5f})")
        out.append(f" Pearson Chi-Square = {chi.pearson:.5f} (with df = {chi.df}; p = {chi.p_pearson:.5f})")
        out.append(f"   for cells with fitted frequencies greater than {chi.threshold:.2f}")
        out.append("   (df = number of cells - 1 - number of moments fit)")
        out.append("")
    out.append(f" Reliability (from above moments) = {r.reliability_from_moments:10.5f}")
    out.append(f" SEM (from above moments) = {r.sem_from_moments:10.5f}")
    out += ["", " ***Numerical Integration***", "", " " + r.quad.description(), ""]
    if r.bivariate_sum is not None:
        out += [f" Sum of bivariate probabilities = {r.bivariate_sum:15.5f}", ""]
    out.append("")
    for table in r.tables:
        out += _table(table)

    fb = r.feedback
    if fb.warnings or fb.notes:
        out += [RULE, "", " ***Warnings and Notes***", ""]
        out += [f" WARNING: {w}" for w in fb.warnings]
        out += [f" NOTE: {n}" for n in fb.notes]
        out.append("")
    return "\n".join(out) + "\n"


def true_distribution(r: Results) -> str:
    lines = ["    midpoint        height      interval           cdf"]
    for row in r.true_distribution:
        lines.append("".join(f"{v:14.8f}" for v in row))
    return "\n".join(lines) + "\n"


def observed_distribution(r: Results) -> str:
    fitted = r.fitted_pmf
    n_ex = r.n_examinees
    lines = ["  score   raw prop   fitted prop      raw freq   fitted freq"]
    for i in range(r.n + 1):
        if r.observed_counts is not None:
            raw_freq = r.observed_counts[i]
            raw = f"{raw_freq / n_ex:11.6f}"
            raw_f = f"{raw_freq:13.4f}"
        else:
            raw, raw_f = f"{'NA':>11}", f"{'NA':>13}"
        lines.append(f"{i:7d}{raw}{fitted[i]:14.6f}{raw_f}{fitted[i] * n_ex:14.4f}")
    return "\n".join(lines) + "\n"


def output_paths(cards_path: Path) -> dict[str, Path]:
    stem = cards_path.name
    folder = cards_path.parent
    return {"out": folder / f"{stem} out.txt",
            "true": folder / f"{stem} true dist.txt",
            "observed": folder / f"{stem} observed dist.txt"}


def write_outputs(r: Results) -> dict[str, Path]:
    paths = output_paths(r.cards.path)
    contents = {"out": main_output(r), "true": true_distribution(r), "observed": observed_distribution(r)}
    for key, path in paths.items():
        path.write_text(contents[key], encoding="utf-8")
    return paths
