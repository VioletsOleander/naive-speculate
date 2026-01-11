"""Provide `DependencyContainer`, as dependency provider for speculative decoding components."""

from functools import cached_property
from typing import TYPE_CHECKING

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from naive_speculate.speculate import SpeculativeDecoder
from naive_speculate.utils.tokenizer import Tokenizer

from .maker import make_drafter, make_inferencer, make_kvcache, make_lm, make_scorer

if TYPE_CHECKING:
    from naive_speculate.config.internal import SpeculateConfig

__all__ = ["DependencyContainer"]


class DependencyContainer:
    """Centralized container for initializing and managing all necessary dependencies.

    Specifically, it handles the initialization of the tokenizer and the speculative decoder.

    Attributes:
        config (SpeculateConfig): Configurations used for initializing dependencies.
    """

    def __init__(self, config: SpeculateConfig) -> None:
        self.config = config

    @cached_property
    def tokenizer(self) -> Tokenizer:
        """The initialized tokenizer instance."""
        hf_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            self.config.draft.infer.model_name,
        )
        return Tokenizer(tokenizer=hf_tokenizer)

    @cached_property
    def speculative_decoder(self) -> SpeculativeDecoder:
        """The initialized speculative decoder instance."""
        draft_lm = make_lm(model_name=self.config.draft.infer.model_name)
        draft_inferencer = make_inferencer(
            language_model=draft_lm,
            inferencer_type=self.config.draft.infer.inferencer_type,
        )
        drafter = make_drafter(inferencer=draft_inferencer)

        score_lm = make_lm(model_name=self.config.verify.infer.model_name)
        scorer = make_scorer(language_model=score_lm)

        drafter_kvcache = make_kvcache(kvcache_type=self.config.draft.infer.kvcache_type)
        scorer_kvcache = make_kvcache(kvcache_type=self.config.verify.infer.kvcache_type)

        return SpeculativeDecoder(
            drafter=drafter,
            scorer=scorer,
            drafter_kvcache=drafter_kvcache,
            scorer_kvcache=scorer_kvcache,
        )
