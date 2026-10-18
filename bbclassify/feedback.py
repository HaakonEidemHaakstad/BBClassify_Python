"""Errors and run-time feedback (warnings and notes) shown to the user."""

from __future__ import annotations

from dataclasses import dataclass, field


class BBClassifyError(Exception):
    """An error that stops the analysis.

    The message is written for the person running the program: it says what
    is wrong, where (control card, file, line), and how to fix it.
    """


@dataclass
class Feedback:
    """Warnings and notes collected during a run.

    Warnings flag results that may be unreliable or inputs that look
    suspicious. Notes record decisions the program made on the user's behalf
    (defaults, fall-backs). Both are printed to the console and written to the
    main output file.
    """

    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)
