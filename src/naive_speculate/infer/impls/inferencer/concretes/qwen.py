from functools import cached_property
from typing import TYPE_CHECKING, override

from transformers import Qwen3ForCausalLM

from naive_speculate.infer.impls.inferencer.abstracts import ForwardOutput
from naive_speculate.infer.impls.inferencer.abstracts.chunkwise import ChunkwiseDecodeInferencer
from naive_speculate.infer.impls.kvcache.dynamic_cache import DynamicNoUpdateCache

if TYPE_CHECKING:
    import torch

    from naive_speculate.infer import KVCache


class QwenInferencer(ChunkwiseDecodeInferencer):
    """Inferencer for Qwen models with dynamic kv cache support.

    Attributes:
        qwen_model (Qwen3ForCausalLM): The Qwen model for causal language modeling.
    """

    qwen_model: Qwen3ForCausalLM

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.qwen_model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            local_files_only=True,
        )

    @cached_property
    @override
    def _eos_token_id(self) -> int:
        """End-of-sequence token id.

        Raises:
            ValueError: If the model config does not have an eos_token_id.
        """
        eos_token_id = self.qwen_model.config.eos_token_id
        if eos_token_id is None:
            raise ValueError("The model config does not have an eos_token_id.")
        return eos_token_id

    @override
    def _forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> ForwardOutput:
        """Forward the model with `query_token_ids`.

        IMPORTANT: This method update the kv cache internally. Therefore,
        expect the `update` method of `kv_cache` to be no-op.

        Return the logits output from the model.

        Args:
            query_token_ids (torch.Tensor): Query token ids of shape `[batch_size, num_query_tokens]`.
            kv_cache (KVCache): Past key value tensors.

        Returns:
            torch.Tensor: The logits output from the model of shape
                `[batch_size, num_query_tokens, vocab_size]`.

        Raises:
            TypeError: If the `kv_cache` is not of type `DynamicNoUpdateCache`.
            ValueError: If the model forward output does not contain logits.
        """
        if not isinstance(kv_cache, DynamicNoUpdateCache):
            raise TypeError(
                f"Expected kv_cache to be of type DynamicNoUpdateCache, but got {type(kv_cache)}."
            )

        input_ids = query_token_ids
        forward_out = self.qwen_model.forward(
            input_ids=input_ids,  # type: ignore[arg-type]
            logits_to_keep=0,  # keeps all logits
            use_cache=True,
            past_key_values=kv_cache.cache,
        )

        if forward_out.logits is None:
            raise ValueError("Model forward output does not contain logits.")

        return ForwardOutput._make((forward_out.logits, (), ()))
