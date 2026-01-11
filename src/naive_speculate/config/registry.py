"""Registry for available implementation types."""

from enum import StrEnum, auto

__all__ = ["InferencerType", "KVCacheType", "LanguageModelType"]


class LanguageModelType(StrEnum):
    """Implemented `LanguageModel` types.

    Each type corresponds to a specific model family.
    (Currently only Qwen3 is supported.)
    """

    QWEN3 = auto()


class InferencerType(StrEnum):
    """Implemented `Inferencer` types."""

    BASIC = auto()
    CHUNKWISE = auto()


class KVCacheType(StrEnum):
    """Implemented `KVCache` types."""

    DYNAMIC = auto()
    DYNAMIC_NO_UPDATE = auto()
