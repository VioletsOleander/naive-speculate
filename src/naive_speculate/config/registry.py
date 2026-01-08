"""Registry for available implementation types."""

from enum import StrEnum, auto


class ModelFamily(StrEnum):
    """Supported model families."""

    QWEN3 = auto()


class InferencerType(StrEnum):
    """Implemented inferencer types."""

    BASIC = auto()
    CHUNKWISE = auto()


class KVCacheType(StrEnum):
    """Implemented KV cache types."""

    DYNAMIC = auto()
    DYNAMIC_NO_UPDATE = auto()
