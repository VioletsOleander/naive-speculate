"""Manages dependencies injection.

Exports:
    DependencyContainer: Contains all assembled dependencies for speculative decoding.
"""

from .container import DependencyContainer

__all__ = ["DependencyContainer"]
