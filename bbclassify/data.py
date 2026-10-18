"""Reading and validating data files (raw scores, frequency distributions, moments)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .controlcards import ControlCards
from .feedback import BBClassifyError
from .model import Moments


@dataclass
class ScoreData:
    scores: np.ndarray | None          # distinct observed scores
    freqs: np.ndarray | None           # frequency (or proportion) of each score
    moments: Moments | None            # moments input (data type M)
    n_examinees: float
    nitems: int | None                 # HB test length
    xmin: float | None                 # LL possible score range
    xmax: float | None
    category_proportions: np.ndarray | None  # moments input only
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _read_records(cards: ControlCards) -> list[tuple[int, list[str]]]:
    try:
        text = cards.data_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise BBClassifyError(f"The data file \"{cards.data_name}\" could not be read: {exc.strerror}.") from None
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    records = [(i + 1, line.split()) for i, line in enumerate(lines) if line.strip()]
    if not records:
        raise BBClassifyError(f"The data file \"{cards.data_name}\" is empty.")
    return records


def _column(records: list[tuple[int, list[str]]], col: int, what: str, name: str) -> np.ndarray:
    values = np.empty(len(records))
    bad = []
    for index, (line_no, fields) in enumerate(records):
        if len(fields) < col:
            raise BBClassifyError(
                f"Line {line_no} of the data file \"{name}\" has {len(fields)} column(s), but control card 2 says the "
                f"{what} are in column {col}.")
        try:
            values[index] = float(fields[col - 1])
        except ValueError:
            bad.append(f"line {line_no}: '{fields[col - 1]}'")
            continue
        if not math.isfinite(values[index]):
            bad.append(f"line {line_no}: '{fields[col - 1]}'")
    if bad:
        shown = "; ".join(bad[:5]) + (f"; and {len(bad) - 5} more" if len(bad) > 5 else "")
        raise BBClassifyError(
            f"The data file \"{name}\" has non-numeric {what} ({shown}). The file must contain only numbers; "
            "remove any header line or labels.")
    return values


def read_data(cards: ControlCards) -> ScoreData:
    records = _read_records(cards)
    name = cards.data_name
    notes: list[str] = []
    warnings: list[str] = []

    if cards.data_type == "M":
        return _read_moments(cards, records)

    if cards.data_type == "R":
        raw = _column(records, cards.xcol, "scores", name)
        if len(records) == 1 and len(records[0][1]) > 1:
            warnings.append(
                f"The data file \"{name}\" has a single line with {len(records[0][1])} values, but only column {cards.xcol} "
                "was read, giving one examinee. Raw scores must be one examinee per line.")
        scores, freqs = np.unique(raw, return_counts=True)
        freqs = freqs.astype(float)
    else:
        score_col = _column(records, cards.xcol, "scores", name)
        freq_col = _column(records, cards.fcol, "frequencies", name)
        if np.any(freq_col < 0):
            line = records[int(np.argmax(freq_col < 0))][0]
            raise BBClassifyError(f"The data file \"{name}\" has a negative frequency (line {line}).")
        if freq_col.sum() <= 0:
            raise BBClassifyError(f"The frequencies in the data file \"{name}\" sum to zero.")
        if len(np.unique(score_col)) < len(score_col):
            warnings.append(f"Some scores appear on more than one line of the data file \"{name}\"; their frequencies were added.")
        scores, inverse = np.unique(score_col, return_inverse=True)
        freqs = np.bincount(inverse, weights=freq_col)
        if np.any(freq_col != np.round(freq_col)) and abs(freq_col.sum() - 1) < 1e-6:
            notes.append("The frequencies sum to 1 and were treated as proportions, so the reported number of examinees is 1.")

    nitems = None
    if cards.procedure == "HB":
        if cards.data_type == "R":
            nitems = cards.nitems
        else:
            # As in BB-CLASS: the frequency distribution has one record per score 0..n.
            nitems = len(records) - 1
            expected = np.arange(nitems + 1)
            if len(score_col) != len(expected) or not np.array_equal(np.sort(score_col), expected):
                raise BBClassifyError(
                    f"For HB with a frequency distribution, the data file must have one line for every possible score from 0 "
                    f"to the number of items, in order (so the number of items is the number of lines minus 1). "
                    f"\"{name}\" has {len(records)} lines with scores from {score_col.min():g} to {score_col.max():g}. "
                    "Add lines with frequency 0 for scores no one obtained.")
            if nitems < 2:
                raise BBClassifyError("For HB the test must have at least 2 items.")
        nonint = scores[scores != np.round(scores)]
        if len(nonint):
            raise BBClassifyError(
                f"For HB the scores must be numbers of items correct (whole numbers), but the data contain "
                f"{', '.join(f'{v:g}' for v in nonint[:5])}. Use LL for scores that are not numbers correct.")
    elif np.any(scores != np.round(scores)):
        notes.append("Some scores are not whole numbers. The LL procedure is designed for reported scores rounded to integers; "
                     "results are computed as given.")

    keep = freqs > 0
    return ScoreData(scores=scores[keep], freqs=freqs[keep], moments=None, n_examinees=float(freqs.sum()),
                     nitems=nitems, xmin=cards.xmin, xmax=cards.xmax, category_proportions=None,
                     notes=notes, warnings=warnings)


def _read_moments(cards: ControlCards, records: list[tuple[int, list[str]]]) -> ScoreData:
    name = cards.data_name
    tokens = [t for _, fields in records for t in fields]
    expected = 7 + cards.ncat
    layout = ("number of examinees, mean, standard deviation, skewness, kurtosis, minimum score, maximum score, "
              f"and the proportions of examinees in the {cards.ncat} categories")
    if len(tokens) != expected:
        raise BBClassifyError(
            f"The moments file \"{name}\" must contain {expected} numbers ({layout}), but it contains {len(tokens)}.")
    try:
        values = [float(t) for t in tokens]
    except ValueError:
        bad = next(t for t in tokens if not _is_number(t))
        raise BBClassifyError(f"The moments file \"{name}\" contains '{bad}', which is not a number. Expected: {layout}.") from None
    n_examinees, mean, sd, skew, kurt, xmin, xmax = values[:7]
    props = np.array(values[7:])
    notes: list[str] = []
    warnings: list[str] = []
    if n_examinees <= 0:
        raise BBClassifyError(f"The moments file \"{name}\": the number of examinees must be positive.")
    if sd <= 0:
        raise BBClassifyError(f"The moments file \"{name}\": the standard deviation must be positive.")
    if kurt <= 1:
        raise BBClassifyError(
            f"The moments file \"{name}\": the kurtosis must be greater than 1 (it is the fourth standardized moment, "
            "about 3 for a normal distribution, not the excess kurtosis).")
    if not xmax > xmin:
        raise BBClassifyError(f"The moments file \"{name}\": the maximum score must be greater than the minimum score.")
    if not xmin <= mean <= xmax:
        raise BBClassifyError(f"The moments file \"{name}\": the mean ({mean:g}) is outside the score range {xmin:g} to {xmax:g}.")
    if np.any(props < 0):
        raise BBClassifyError(f"The moments file \"{name}\": category proportions cannot be negative.")
    if abs(props.sum() - 1) > 1e-3:
        warnings.append(f"The category proportions in the moments file sum to {props.sum():.5f}, not 1; they were rescaled to sum to 1.")
    props = props / props.sum()
    nitems = None
    if cards.procedure == "HB":
        if xmin != 0 or xmax != round(xmax) or xmax < 2:
            raise BBClassifyError(
                f"The moments file \"{name}\": for HB the minimum score must be 0 and the maximum score must be the number "
                f"of items (a whole number), but {xmin:g} and {xmax:g} were given.")
        nitems = int(xmax)
    notes.append("Moments were used as input, so chi-square statistics are not available.")
    return ScoreData(scores=None, freqs=None, moments=Moments(mean, sd, skew, kurt), n_examinees=n_examinees,
                     nitems=nitems, xmin=xmin, xmax=xmax, category_proportions=props, notes=notes, warnings=warnings)


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True
