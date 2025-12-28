from functools import cached_property
from typing import cast, override

import torch
from transformers import (
    DynamicCache,
    DynamicLayer,
    Qwen3ForCausalLM,
)

from naive_speculate.infer.implementations.chunkwise import ChunkwiseDecodeInferencer


class QwenInferencer(ChunkwiseDecodeInferencer):
    """Inferencer for Qwen models with dynamic kv cache support.

    Attributes:
        qwen_model (Qwen3ForCausalLM): The Qwen model for causal language modeling.
        kv_cache (DynamicCache): The dynamic key-value cache for the model.
    """

    qwen_model: Qwen3ForCausalLM
    kv_cache: DynamicCache

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.qwen_model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            local_files_only=True,
        )
        self.kv_cache = DynamicCache(config=self.qwen_model.config)

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
    def _forward(self, query_token_ids: torch.Tensor) -> torch.Tensor:
        """Forward the model with `query_token_ids`.

        Update the kv cache internally.

        Return the logits output from the model.

        Args:
            query_token_ids (torch.Tensor): Query token ids of shape `[batch_size, num_query_tokens]`.

        Returns:
            torch.Tensor: The logits output from the model of shape `[batch_size, num_query_tokens, vocab_size]`.

        Raises:
            ValueError: If the model forward output does not contain logits.
        """
        input_ids = cast(torch.LongTensor, query_token_ids)
        forward_out = self.qwen_model.forward(
            input_ids=input_ids,
            logits_to_keep=0,  # keeps all logits
            use_cache=True,
            past_key_values=self.kv_cache,
        )

        if forward_out.logits is None:
            raise ValueError("Model forward output does not contain logits.")

        return forward_out.logits

    # TODO: move these dirty debugging methods to testing module, using monkey patching
    # or other techniques during testing only.
    def _reset(self) -> None:
        """Reset the model state for a new inference session.

        Primarily used for testing purpose.
        """
        self.kv_cache.crop(0)

    def _print_kvcache_shape(self) -> None:
        """Print model's kv cache shape.

        Primarily used for debugging purpose.
        It is assumed that all layers have the same kv cache shape.
        """
        layer = cast(DynamicLayer, self.kv_cache.layers[0])
        if layer.keys is not None:
            print(f"Keys shape: {layer.keys.shape}", end=", ")
        else:
            print("Keys shape: None", end=", ")

        if layer.values is not None:
            print(f"Values shape: {layer.values.shape}")
        else:
            print("Values shape: None")
