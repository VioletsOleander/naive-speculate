import json
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from naive_speculate.speculate import VerifyStrategy
from naive_speculate.utils.config import SpeculateConfig
from naive_speculate.utils.sample import SampleStrategy
from naive_speculate.utils.tokenizer import Tokenizer

if TYPE_CHECKING:
    from naive_speculate.infer import Inferencer
    from naive_speculate.speculate import SpeculativeDecoder


class SupportedModelFamilies(StrEnum):
    QWEN3 = "Qwen3"


class SupportedInferencerTypes(StrEnum):
    BASIC = "Basic"
    CHUNKWISE = "Chunkwise"


class DependencyContainer:
    """Centralized container for initializing and managing all necessary dependencies for speculative decoding."""

    def __init__(self, config_path: str, context_path: str) -> None:
        self.config_path = config_path
        self.context_path = context_path

        self._init_inferencers()
        self._init_drafter()
        self._init_scorer()
        self._init_kvcache()
        self._init_speculative_decoder()

    @cached_property
    def speculate_config(self) -> SpeculateConfig:
        """Configuration loaded from the specified config file."""
        return SpeculateConfig.from_file(self.config_path)

    @cached_property
    def draft_strategy(self) -> SampleStrategy:
        """Sampling strategy for drafting."""
        return self.speculate_config.sample_strategy

    @cached_property
    def verify_strategy(self) -> VerifyStrategy:
        """Sampling strategy for the verification."""
        match self.draft_strategy:
            case SampleStrategy.GREEDY:
                return VerifyStrategy.GREEDY_MATCH
            case SampleStrategy.RANDOM:
                return VerifyStrategy.SPECULATIVE_SAMPLE
            case _:
                raise ValueError(f"Unsupported draft strategy: {self.draft_strategy}")

    @cached_property
    def context(self) -> list[dict[str, str]]:
        """Input context loaded from the specified context file."""
        with Path(self.context_path).open("r", encoding="utf-8") as f:
            context: list[dict[str, str]] = json.load(f)

        return context

    @cached_property
    def tokenizer(self) -> Tokenizer:
        """The initialized tokenizer instance."""
        hf_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            self.speculate_config.drafter_model_name
        )

        return Tokenizer(tokenizer=hf_tokenizer)

    @property
    def speculative_decoder(self) -> SpeculativeDecoder:
        """The initialized speculative decoder instance."""
        return self._speculative_decoder

    def _init_inferencers(self) -> None:
        def provide_inferencer(
            model_name: str,
            inferencer_type: SupportedInferencerTypes = SupportedInferencerTypes.CHUNKWISE,
        ) -> Inferencer:
            model_family = model_name.split("/")[0]
            if model_family not in SupportedModelFamilies:
                raise ValueError(f"Unsupported model family: {model_family}")

            match model_family:
                case SupportedModelFamilies.QWEN3:
                    import naive_speculate.infer.impls.inferencer.concretes.qwen3 as impl_module  # noqa: PLC0415

            match inferencer_type:
                case SupportedInferencerTypes.BASIC:
                    inferencer_class = impl_module.BasicInferencerImpl
                case SupportedInferencerTypes.CHUNKWISE:
                    inferencer_class = impl_module.ChunkwiseInferencerImpl

            return inferencer_class(model_name=model_name)

        self._drafter_inferencer = provide_inferencer(
            model_name=self.speculate_config.drafter_model_name
        )
        self._scorer_inferencer = provide_inferencer(
            model_name=self.speculate_config.verifier_model_name
        )

    def _init_drafter(self) -> None:
        import naive_speculate.draft.impls.drafter as impl_module  # noqa: PLC0415

        draft_class = impl_module.Drafter
        self._drafter = draft_class(inferencer=self._drafter_inferencer)

    def _init_scorer(self) -> None:
        import naive_speculate.score.impls.scorer as impl_module  # noqa: PLC0415

        scorer_class = impl_module.Scorer
        self._scorer = scorer_class(inferencer=self._scorer_inferencer)

    def _init_kvcache(self) -> None:
        import naive_speculate.infer.impls.kvcache.dynamic_cache as impl_module  # noqa: PLC0415

        cache_class = impl_module.DynamicNoUpdateCache
        self._drafter_kv_cache = cache_class()
        self._scorer_kv_cache = cache_class()

    def _init_speculative_decoder(self) -> None:
        import naive_speculate.speculate.speculative_decoder as impl_module  # noqa: PLC0415

        speculative_decoder_class = impl_module.SpeculativeDecoder
        self._speculative_decoder = speculative_decoder_class(
            speculate_config=self.speculate_config,
            drafter=self._drafter,
            scorer=self._scorer,
            drafter_kvcache=self._drafter_kv_cache,
            scorer_kvcache=self._scorer_kv_cache,
        )
