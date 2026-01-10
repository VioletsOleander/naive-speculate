"""Provide Qwen3 model implementation of `LanguageModel`."""

from functools import cached_property
from typing import TYPE_CHECKING, cast, override

from transformers import Qwen3ForCausalLM

from naive_speculate.infer.kvcache.dynamic_no_update import DynamicCache

from .interface import LanguageModel

if TYPE_CHECKING:
    import torch

    from naive_speculate.infer import KVCache

__all__ = ["LanguageModelImpl"]


class Qwen3Model(LanguageModel):
    """Qwen3Model wraps huggingface Qwen3 models, implementing `LanguageModel`.

    Attributes:
        hf_model (Qwen3ForCausalLM): The underlying huggingface Qwen3 model.
    """

    hf_model: Qwen3ForCausalLM

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.hf_model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )

    @cached_property
    @override
    def eos_token_id(self) -> int:
        eos_token_id = self.hf_model.config.eos_token_id
        if eos_token_id is None:
            raise ValueError("The model config does not have an eos_token_id.")

        return eos_token_id

    @override
    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        """Forward the underlying huggingface model.

        Expects `kv_cache` to be an instance of `DynamicCache`.

        Refers to `LanguageModel.forward` for more details.

        Raises:
            ValueError: If the model forward output does not contain logits.
        """
        if not isinstance(kv_cache, DynamicCache):
            raise TypeError("Qwen3Model only supports DynamicCache as KVCache.")

        input_ids = query_token_ids
        forward_out = self.hf_model.forward(
            input_ids=cast("torch.LongTensor", input_ids),
            logits_to_keep=0,  # keeps all logits
            use_cache=True,
            past_key_values=kv_cache.cache,
        )

        if forward_out.logits is None:
            raise ValueError("Model forward output does not contain logits.")

        return forward_out.logits


LanguageModelImpl = Qwen3Model
