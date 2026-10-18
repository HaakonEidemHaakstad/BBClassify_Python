"""Console program: prompts for a control-card file, like BB-CLASS."""

from __future__ import annotations

import os
import sys
import traceback
import warnings
from pathlib import Path

from . import __version__
from .feedback import BBClassifyError

ISSUES_URL = "https://github.com/HaakonEidemHaakstad/BBClassify_Python/issues"


class Console:
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        isatty = getattr(self.stream, "isatty", None)
        self.color = bool(isatty and isatty()) and os.environ.get("NO_COLOR") is None
        if self.color and os.name == "nt":
            os.system("")  # enables ANSI colour codes in the Windows console

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def print(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def error(self, text: str) -> None:
        self.print(self._paint("ERROR: ", "1;91") + text)

    def warning(self, text: str) -> None:
        self.print(self._paint("WARNING: ", "1;93") + text)

    def note(self, text: str) -> None:
        self.print(self._paint("NOTE: ", "1;96") + text)

    def ok(self, text: str) -> None:
        self.print(self._paint(text, "1;92"))


def _find_control_cards(name: str) -> Path:
    from .controlcards import resolve_file
    return resolve_file(name.strip().strip('"'), None, "control-card file")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    interactive = not argv
    console = Console()
    warnings.simplefilter("ignore")  # numerical warnings are handled and reported by the program itself
    console.print(f"BBClassify {__version__}: Beta-Binomial Classification Consistency and Accuracy")
    console.print()
    status = 1
    try:
        if argv:
            name = argv[0]
        else:
            name = input("Enter the name of the file containing the control cards: ")
            if not name.strip():
                raise BBClassifyError("No file name was entered.")
        status = run_file(_find_control_cards(name), console)
    except BBClassifyError as exc:
        console.print()
        console.error(str(exc))
        console.print()
        console.print("Execution terminated. Correct the problem above and run BBClassify again.")
    except KeyboardInterrupt:
        console.print()
        console.print("Execution cancelled.")
    except Exception:  # pragma: no cover - unexpected failures
        log = Path.cwd() / "BBClassify error log.txt"
        try:
            log.write_text(traceback.format_exc(), encoding="utf-8")
            where = f"Details were saved to \"{log}\". "
        except OSError:
            where = ""
        console.print()
        console.error("BBClassify stopped because of an unexpected internal error. This is a bug in BBClassify, not in "
                      f"your input. {where}Please report it, with your control cards and data if possible, at {ISSUES_URL}")
    if interactive:
        console.print()
        try:
            input("Press ENTER to exit.")
        except EOFError:
            pass
    return status


def run_file(path: Path, console: Console) -> int:
    from .analysis import analyze
    from .controlcards import read_control_cards
    from .data import read_data
    from .report import write_outputs

    console.print(f"Control cards in \"{path}\":")
    try:
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if line.strip():
                console.print("  " + line.rstrip())
    except OSError:
        pass
    console.print()
    console.print("Reading control cards...")
    cards = read_control_cards(path)
    console.print(f"Reading data file \"{cards.data_path.name}\"...")
    data = read_data(cards)
    procedure = "Hanson and Brennan" if cards.procedure == "HB" else "Livingston and Lewis"
    console.print(f"Running the {procedure} procedure...")
    results = analyze(cards, data)
    console.print("Writing output files...")
    try:
        paths = write_outputs(results)
    except OSError as exc:
        raise BBClassifyError(f"The output files could not be written to \"{path.parent}\": {exc.strerror}. "
                              "Check that the folder is not read-only and that the files are not open in another program.") from None
    console.print()
    for w in results.feedback.warnings:
        console.warning(w)
    for n in results.feedback.notes:
        console.note(n)
    if results.feedback.warnings or results.feedback.notes:
        console.print()
    main_table = results.tables[0]
    consistency = results.tables[1]
    console.print(f"Probability of correct classification (accuracy): {main_table.correct:.5f}")
    console.print(f"Probability of consistent classification (pc):    {consistency.correct:.5f} (kappa = {consistency.kappa:.5f})")
    console.print()
    console.print("Output files:")
    for p in paths.values():
        console.print(f"  {p}")
    console.print()
    console.ok("Successful execution" + (" (with warnings; see above)." if results.feedback.warnings else "."))
    return 0
