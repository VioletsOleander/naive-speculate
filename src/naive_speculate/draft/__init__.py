"""Drafter interfaces and implementations based on the inference basis.

Exports:
    Drafter: Interface for drafting components.
    DraftResult: Data structure representing the result
"""

from .interfaces import Drafter, DraftResult

__all__ = ["DraftResult", "Drafter"]
