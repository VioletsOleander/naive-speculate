from functools import cached_property
from typing import TYPE_CHECKING, cast, override

from transformers import Qwen3ForCausalLM

from naive_speculate.infer.impl.inferencer.abstract.basic import BasicInferencer
from naive_speculate.infer.impl.inferencer.abstract.chunkwise import ChunkwiseDecodeInferencer
from naive_speculate.infer.impl.kvcache.dynamic_no_update_cache import DynamicNoUpdateCache

if TYPE_CHECKING:
    import torch

    from naive_speculate.infer import KVCache


class Qwen3Model:
    """Wraps huggingface Qwen3 models, providing implementation for `eos_token_id` and `forward`.

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

    @property
    def eos_token_id(self) -> int:
        """Id of the end-of-sequence (EOS) token.

        Raises:
            ValueError: If the model config does not have an eos_token_id.
        """
        eos_token_id = self.hf_model.config.eos_token_id
        if eos_token_id is None:
            raise ValueError("The model config does not have an eos_token_id.")
        return eos_token_id

    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        """Forward the model with `query_token_ids`.

        This method updates the kv cache internally. Therefore,
        expect the `update` method of `kv_cache` to be no-op.
        In other words, expect `kv_cache` to be of type `DynamicNoUpdateCache`.

        Refers to the interface `Inferencer.forward` for more details.

        Raises:
            ValueError: If the model forward output does not contain logits.
        """
        if not isinstance(kv_cache, DynamicNoUpdateCache):
            raise TypeError(
                f"Expected kv_cache to be of type DynamicNoUpdateCache, but got {type(kv_cache)}."
            )

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


class Qwen3BasicInferencer(BasicInferencer):
    """Basic inferencer for Qwen3 models with dynamic kv cache support.

    Fully implements the `Inferencer` protocol.

    Delegates the implementation of `forward` and `_eos_token_id` to the
    underlying `Qwen3Model`.

    Refers to base class `BasicInferencer` for more details.

    Attributes:
        qwen3_model (Qwen3Model): The underlying Qwen3 model.
    """

    qwen3_model: Qwen3Model

    def __init__(self, model: Qwen3Model) -> None:
        self.qwen3_model = model

    @cached_property
    @override
    def _eos_token_id(self) -> int:
        return self.qwen3_model.eos_token_id

    @override
    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        return self.qwen3_model.forward(query_token_ids, kv_cache)


class Qwen3ChunkwiseInferencer(ChunkwiseDecodeInferencer):
    """Chunkwise inferencer for Qwen3 models with dynamic kv cache support.

    Fully implements the `Inferencer` protocol.

    Delegates the implementation of `forward` and `_eos_token_id` to the
    underlying `Qwen3Model`.

    Refers to base class `ChunkwiseDecodeInferencer` for more details.

    Attributes:
        qwen3_model (Qwen3Model): The underlying Qwen3 model.
    """

    qwen3_model: Qwen3Model
    _decode_chunk_size: int = 8

    def __init__(self, model: Qwen3Model) -> None:
        self.qwen3_model = model

    @property
    @override
    def decode_chunk_size(self) -> int:
        return self._decode_chunk_size

    @cached_property
    @override
    def _eos_token_id(self) -> int:
        return self.qwen3_model.eos_token_id

    @override
    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        return self.qwen3_model.forward(query_token_ids, kv_cache)


Model = Qwen3Model
BasicInferencerImpl = Qwen3BasicInferencer
ChunkwiseInferencerImpl = Qwen3ChunkwiseInferencer
