"""Provide `SampleStrategy` and `VerifyStrategy` for token generation and verification."""

from enum import StrEnum, auto

__all__ = ["SampleStrategy", "VerifyStrategy"]


class SampleStrategy(StrEnum):
    """Sampling strategies for token generation.

    Attributes:
        RANDOM: Sample tokens probabilistically according to the token distribution over vocabulary.
        GREEDY: Always select the token with the highest probability (argmax).
    """

    RANDOM = auto()
    GREEDY = auto()


class VerifyStrategy(StrEnum):
    """Verification strategies for speculative decoding.

    Attributes:
        GREEDY_MATCH: Verify drafted tokens using greedy matching.
        SPECULATIVE_SAMPLING: Verify drafted tokens using speculative sampling.
    """

    GREEDY_MATCH = auto()
    SPECULATIVE_SAMPLING = auto()
