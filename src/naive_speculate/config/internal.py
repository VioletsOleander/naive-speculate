"""Framework facing interface, defining internal configuration options for dependency manager."""

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from naive_speculate.config.strategy import SampleStrategy, VerifyStrategy

    from .registry import InferencerType, KVCacheType

__all__ = ["DraftConfig", "InferenceConfig", "SpeculateConfig", "VerifyConfig"]


class InferenceConfig(BaseModel):
    """Configuration for assembling an inferencer.

    Attributes:
        model_name (str): HuggingFace model name.
        kvcache_type (KVCacheType): The type of key-value cache to be used.
        inferencer_type (InferencerType): The type of inferencer to be used.
    """

    model_name: str
    kvcache_type: KVCacheType
    inferencer_type: InferencerType


class DraftConfig(BaseModel):
    """Configuration for setting up a drafter.

    Attributes:
        sample_strategy (SampleStrategy): The sampling strategy for drafting.
        num_draft_tokens (int): Number of tokens to draft in each iteration.
        infer (InferenceConfig): Configuration about inference.
    """

    sample_strategy: SampleStrategy
    num_draft_tokens: int
    infer: InferenceConfig


class VerifyConfig(BaseModel):
    """Configuration for setting up a verifier.

    Attributes:
        verify_strategy (VerifyStrategy): The verification strategy.
        infer (InferenceConfig): Configuration about inference.
    """

    verify_strategy: VerifyStrategy
    infer: InferenceConfig


class SpeculateConfig(BaseModel):
    """Configuration for setting up a speculative decoder.

    Attributes:
        draft (DraftConfig): Configuration for the drafter.
        verify (VerifyConfig): Configuration for the verifier.
    """

    draft: DraftConfig
    verify: VerifyConfig
