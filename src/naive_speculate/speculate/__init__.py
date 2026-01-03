"""Speculative decoding implementations based on Drafter and Scorer interfaces.

Exports:
    SpeculativeDecoder: Entry class for speculative decoding.
    VerifyStrategy: Enum for verification strategies.
"""

from .speculative_decoder import SpeculativeDecoder, VerifyStrategy

__all__ = ["SpeculativeDecoder", "VerifyStrategy"]
