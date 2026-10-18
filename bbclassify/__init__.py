"""BBClassify: beta-binomial classification consistency and accuracy.

A corrected re-implementation of BB-CLASS (Brennan, 2004). See README.md.
"""

__version__ = "1.0.0"

from .feedback import BBClassifyError, Feedback  # noqa: E402

__all__ = ["BBClassifyError", "Feedback", "run", "__version__"]


def run(control_cards: str, write: bool = True):
    """Run an analysis from a control-card file and return the results.

    Output files are written next to the control-card file unless ``write``
    is False.
    """
    from .analysis import analyze
    from .controlcards import read_control_cards
    from .data import read_data
    from .report import write_outputs

    cards = read_control_cards(control_cards)
    results = analyze(cards, read_data(cards))
    if write:
        results.output_files = write_outputs(results)
    return results
