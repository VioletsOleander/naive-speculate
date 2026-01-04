from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from naive_speculate.infer import DecodeOutput

from .basic import BasicInferencer
from .utils.collection import OutputCollection

if TYPE_CHECKING:
    from naive_speculate.infer import KVCache
    from naive_speculate.utils.sample import SampleStrategy


class ChunkwiseDecodeInferencer(BasicInferencer):
    """ChunkwiseDecodeInferencer implements chunk-wise decoding to reduce device synchronization overhead.

    ChunkwiseDecodeInferencer only checks eos token after each `decode_chunk_size` iterations.

    ChunkwiseDecodeInferencer expects inheriting classes to implement the following additional abstract methods:
    - `decode_chunk_size`: Return the number of tokens to decode before checking for EOS token.

    Refers to base class `BaseInferencer` for more details.
    """

    @property
    @abstractmethod
    def decode_chunk_size(self) -> int:
        """Number of tokens to decode before checking for EOS token.

        Used as a simple trick to reduce device synchronization overhead.
        """
        ...

    @torch.no_grad()
    def decode(
        self,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
        max_new_tokens: int,
        sample_strategy: SampleStrategy,
    ) -> DecodeOutput:
        """Process `query_token_ids` and generate new tokens, auto-regressively repeat.

        Check for EOS token after each `self.decode_chunk_size` generation iterations.

        Refers to the interface `Inferencer.decode` for more details.
        """
        output_collection = OutputCollection()

        if max_new_tokens <= 0:
            return DecodeOutput._make(output_collection.finalize())

        stream = self._generation_stream(
            query_token_ids=query_token_ids, kv_cache=kv_cache, sample_strategy=sample_strategy
        )

        num_new_tokens = 0
        chunk_size = self.decode_chunk_size
        max_chunks = (max_new_tokens + chunk_size - 1) // chunk_size

        for _ in range(max_chunks):
            decode_chunk_size = min(chunk_size, max_new_tokens - num_new_tokens)

            # 1. Decode `decode_chunk_size` tokens continuously
            for _ in range(decode_chunk_size):
                output_ids, output_logits = next(stream)
                output_collection.update(output_ids, output_logits)
            num_new_tokens += decode_chunk_size

            # 2. Check for EOS token existence in the last chunk
            eos_token_idx = output_collection.find(
                self._eos_token_id, start_idx=num_new_tokens - decode_chunk_size
            )

            if eos_token_idx != -1:
                num_excess_tokens = num_new_tokens - (eos_token_idx + 1)
                return DecodeOutput._make(
                    output_collection.finalize(num_tokens_trim=num_excess_tokens)
                )

        return DecodeOutput._make(output_collection.finalize())
