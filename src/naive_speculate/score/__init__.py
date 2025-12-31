"""Scorer interfaces and implementations based on the inference basis.

Exports:
    Scorer: Interface for scoring mechanisms.
    ScoreResult: Data structure for holding scoring results.
"""

from .interfaces import Scorer, ScoreResult

__all__ = ["ScoreResult", "Scorer"]
