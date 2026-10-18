"""Reading and validating BB-CLASS control cards.

The control-card file has the same format as for BB-CLASS (Brennan, 2004):

Card 1 (procedures):
    type  r  nparm  [xfit a]  [Gquad | EHquad | EDquad]  [npts b]  [check]

Card 2 (input data):
    HB:  "data"  F  xcol  fcol
         "data"  R  xcol  nitems
         "data"  M
    LL:  "data"  F  xcol  fcol  xmax  [xmin]
         "data"  R  xcol  xmax  [xmin]
         "data"  M

Card 3 (cut scores):
    K  xcut[1..K-1]  [tcut[1..K-1]]

The only addition to BB-CLASS is ``xmax`` (required) and ``xmin`` (optional,
default 0) for LL runs with raw or frequency data: the highest and lowest
POSSIBLE scores on the test, which the effective test length requires.
(BB-CLASS used the highest and lowest observed scores instead.) With moments
input (M), the minimum and maximum are read from the moments file, as in
BB-CLASS, and must likewise be the possible scores.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .feedback import BBClassifyError
from .model import QUADRATURE_METHODS

_TOKEN = re.compile(r'"[^"]*"|\S+')
CARD_FORMATS = {
    ("HB", "F"): '"data" F xcol fcol',
    ("HB", "R"): '"data" R xcol nitems',
    ("HB", "M"): '"data" M',
    ("LL", "F"): '"data" F xcol fcol xmax [xmin]',
    ("LL", "R"): '"data" R xcol xmax [xmin]',
    ("LL", "M"): '"data" M',
}


@dataclass
class ControlCards:
    path: Path
    lines: list[str]
    procedure: str
    reliability: float
    nparm: int
    xfit: float
    quadrature: str
    npts: int
    check: bool
    data_name: str
    data_path: Path
    data_type: str
    xcol: int | None
    fcol: int | None
    nitems: int | None
    xmax: float | None
    xmin: float | None
    ncat: int
    xcuts: list[float]
    tcuts: list[float] | None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _tokens(line: str, card: int) -> list[str]:
    if line.count('"') % 2:
        raise BBClassifyError(f"Control card {card} has an unmatched double quotation mark: {line.strip()}")
    return _TOKEN.findall(line)


def _number(token: str, card: int, what: str) -> float:
    try:
        value = float(token)
    except ValueError:
        raise BBClassifyError(f"Control card {card}: {what} must be a number, but '{token}' was given.") from None
    if not math.isfinite(value):
        raise BBClassifyError(f"Control card {card}: {what} must be a finite number, but '{token}' was given.")
    return value


def _integer(token: str, card: int, what: str, minimum: int = 1) -> int:
    value = _number(token, card, what)
    if value != int(value) or value < minimum:
        raise BBClassifyError(
            f"Control card {card}: {what} must be a whole number of at least {minimum}, but '{token}' was given.")
    return int(value)


def app_directory() -> Path:
    """Folder of the running program (the executable when frozen with PyInstaller)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def resolve_file(name: str, relative_to: Path | None, what: str) -> Path:
    """Find a file by absolute path, or relative to the control-card folder,
    the program folder, or the current folder (in that order)."""
    candidate = Path(name).expanduser()
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise BBClassifyError(f"The {what} \"{name}\" was not found.")
    places = []
    for base in (relative_to, app_directory(), Path.cwd()):
        if base is None or base in places:
            continue
        places.append(base)
        if (base / candidate).is_file():
            return (base / candidate).resolve()
    looked = "\n  ".join(str(p) for p in places)
    raise BBClassifyError(
        f"The {what} \"{name}\" was not found. Looked in:\n  {looked}\n"
        "Give the full path, or put the file in one of these folders. Check the spelling, including any spaces "
        "and the file extension (Windows may hide extensions such as .txt).")


def read_control_cards(path: str | Path) -> ControlCards:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise BBClassifyError(f"The control-card file \"{path}\" could not be read: {exc.strerror}.") from None
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line for line in raw_lines if line.strip()]
    notes: list[str] = []
    warnings: list[str] = []
    if len(lines) < 3:
        raise BBClassifyError(
            f"The control-card file \"{path.name}\" must contain three control cards (lines), but {len(lines)} "
            "non-empty line(s) were found. See the BB-CLASS manual or README for the format.")
    if len(lines) > 3:
        warnings.append(f"The control-card file has {len(lines)} non-empty lines; only the first three were used.")
    card1, card2, card3 = (_tokens(line, i + 1) for i, line in enumerate(lines[:3]))

    # -- Card 1 --------------------------------------------------------------
    if len(card1) < 3:
        raise BBClassifyError(
            "Control card 1 must start with the procedure (HB or LL), the reliability and the number of beta "
            f"parameters (2 or 4), e.g. 'HB 0. 4'. Found: {lines[0].strip()}")
    procedure = card1[0].upper()
    if procedure not in ("HB", "LL"):
        raise BBClassifyError(
            f"Control card 1: the procedure must be HB (Hanson and Brennan) or LL (Livingston and Lewis), "
            f"but '{card1[0]}' was given.")
    reliability = _number(card1[1], 1, "the reliability")
    if procedure == "HB" and not 0 <= reliability < 1:
        raise BBClassifyError(
            f"Control card 1: for HB the reliability must be at least 0 and less than 1 ({reliability:g} was given). "
            "Use 0 for the binomial error model, or the test's reliability for the compound binomial model.")
    if procedure == "LL" and not 0 < reliability < 1:
        raise BBClassifyError(
            f"Control card 1: for LL the reliability must be greater than 0 and less than 1 ({reliability:g} was given), "
            "because it is needed to estimate the effective test length.")
    nparm = _number(card1[2], 1, "the number of beta parameters")
    if nparm not in (2, 4):
        raise BBClassifyError(
            f"Control card 1: the number of beta parameters must be 2 or 4, but '{card1[2]}' was given.")
    nparm = int(nparm)

    xfit, quad, npts, check = 0.0, None, None, False
    i = 3
    options = {m.lower(): m for m in QUADRATURE_METHODS}
    while i < len(card1):
        token = card1[i].lower()
        if token == "xfit":
            if i + 1 >= len(card1):
                raise BBClassifyError("Control card 1: 'xfit' must be followed by a number, e.g. 'xfit 5'.")
            xfit = _number(card1[i + 1], 1, "the xfit value")
            if xfit < 0:
                raise BBClassifyError("Control card 1: the xfit value must be at least 0.")
            i += 2
        elif token == "npts":
            if i + 1 >= len(card1):
                raise BBClassifyError("Control card 1: 'npts' must be followed by a whole number, e.g. 'npts 1000'.")
            npts = _integer(card1[i + 1], 1, "the number of quadrature points (npts)", minimum=10)
            i += 2
        elif token in options:
            if quad is not None:
                raise BBClassifyError("Control card 1: specify only one of Gquad, EHquad and EDquad.")
            quad = options[token]
            i += 1
        elif token == "check":
            check = True
            i += 1
        else:
            raise BBClassifyError(
                f"Control card 1: '{card1[i]}' is not a recognized option. After the procedure, reliability and number "
                "of beta parameters, the optional fields are: xfit <number>, Gquad, EHquad, EDquad, npts <number>, check.")
    if quad is None:
        quad = "EDquad"
    if npts is not None and quad == "Gquad":
        notes.append("npts is ignored with Gquad, which always uses 64 points.")
    npts = npts or 1000

    # -- Card 2 --------------------------------------------------------------
    if len(card2) < 2:
        raise BBClassifyError(
            "Control card 2 must give the data file name in double quotation marks and the data type (F, R or M), "
            f"e.g. '\"mydata\" F 1 2'. Found: {lines[1].strip()}")
    data_name = card2[0]
    if data_name.startswith('"'):
        data_name = data_name[1:-1]
    else:
        notes.append(f"The data file name on control card 2 is not in double quotation marks; read it as \"{data_name}\".")
    data_type = card2[1].upper()
    if data_type not in ("F", "R", "M"):
        raise BBClassifyError(
            f"Control card 2: the data type must be F (frequency distribution), R (raw scores) or M (moments), "
            f"but '{card2[1]}' was given.")
    fields = card2[2:]
    usage = CARD_FORMATS[(procedure, data_type)]
    xcol = fcol = nitems = None
    xmax = xmin = None

    def need(count: int, names: str) -> None:
        if len(fields) < count:
            message = f"Control card 2: for {procedure} with data type {data_type}, give {names}. The format is: {usage}"
            if procedure == "LL" and data_type in ("F", "R"):
                message += (
                    "\nUnlike BB-CLASS, BBClassify needs the maximum POSSIBLE score on the test (xmax) and, if it is not 0, "
                    "the minimum POSSIBLE score (xmin). The effective test length depends on them; BB-CLASS used the "
                    "observed maximum and minimum instead, which biases the results.")
            raise BBClassifyError(message)

    if data_type == "F":
        need(2, "the score column and the frequency column")
        xcol = _integer(fields[0], 2, "the score column (xcol)")
        fcol = _integer(fields[1], 2, "the frequency column (fcol)")
        if xcol == fcol:
            raise BBClassifyError("Control card 2: the score column and the frequency column must be different.")
        rest = fields[2:]
    elif data_type == "R":
        need(2, "the score column and " + ("the number of items" if procedure == "HB"
                                                                         else "the maximum possible score (xmax)"))
        xcol = _integer(fields[0], 2, "the score column (xcol)")
        rest = fields[1:]
        if procedure == "HB":
            nitems = _integer(rest[0], 2, "the number of items (nitems)", minimum=2)
            rest = rest[1:]
    else:
        rest = fields
    if procedure == "LL" and data_type in ("F", "R"):
        need((2 if data_type == "F" else 1) + 1, "the maximum possible score (xmax)")
        xmax = _number(rest[0], 2, "the maximum possible score (xmax)")
        if len(rest) > 1:
            xmin = _number(rest[1], 2, "the minimum possible score (xmin)")
            rest = rest[2:]
        else:
            xmin = 0.0
            notes.append("The minimum possible score (xmin) was not given on control card 2, so it was set to 0.")
            rest = []
        if not xmax > xmin:
            raise BBClassifyError(
                f"Control card 2: the maximum possible score ({xmax:g}) must be greater than the minimum possible score ({xmin:g}).")
    if rest:
        warnings.append(f"Extra values on control card 2 were ignored: {' '.join(rest)}. The format is: {usage}")
    data_path = resolve_file(data_name, path.parent, "data file named on control card 2")
    if data_path.resolve() == path.resolve():
        raise BBClassifyError("The data file must be different from the control-card file.")

    # -- Card 3 --------------------------------------------------------------
    if not card3:
        raise BBClassifyError("Control card 3 must give the number of categories and the cut scores.")
    ncat = _integer(card3[0], 3, "the number of categories (K)", minimum=2)
    values = [_number(t, 3, "each cut score") for t in card3[1:]]
    if len(values) == ncat - 1:
        xcuts, tcuts = values, None
    elif len(values) == 2 * (ncat - 1):
        xcuts, tcuts = values[:ncat - 1], values[ncat - 1:]
    else:
        raise BBClassifyError(
            f"Control card 3: with K = {ncat} categories, give {ncat - 1} observed cut score(s), optionally followed by "
            f"{ncat - 1} true cut score(s), but {len(values)} value(s) were given.")
    if any(b <= a for a, b in zip(xcuts, xcuts[1:])):
        raise BBClassifyError("Control card 3: the observed cut scores must be in increasing order.")
    if tcuts is not None:
        if any(not 0 < t < 1 for t in tcuts):
            raise BBClassifyError(
                "Control card 3: true cut scores are proportion-correct scores and must be between 0 and 1, "
                f"but {', '.join(f'{t:g}' for t in tcuts)} were given.")
        if any(b <= a for a, b in zip(tcuts, tcuts[1:])):
            raise BBClassifyError("Control card 3: the true cut scores must be in increasing order.")

    return ControlCards(path=path, lines=[line.rstrip() for line in lines[:3]], procedure=procedure,
                        reliability=reliability, nparm=nparm, xfit=xfit, quadrature=quad, npts=npts, check=check,
                        data_name=data_name, data_path=data_path, data_type=data_type, xcol=xcol, fcol=fcol,
                        nitems=nitems, xmax=xmax, xmin=xmin, ncat=ncat, xcuts=xcuts, tcuts=tcuts,
                        notes=notes, warnings=warnings)
