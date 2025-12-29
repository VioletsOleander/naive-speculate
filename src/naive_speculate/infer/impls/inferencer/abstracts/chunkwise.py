from typing import TYPE_CHECKING

import torch

from naive_speculate.infer import DecodeOutput

from .base import BaseInferencer
from .utils.collection import OutputCollection

if TYPE_CHECKING:
    from naive_speculate.infer import KVCache
    from naive_speculate.utils.sample import SampleStrategy


class ChunkwiseDecodeInferencer(BaseInferencer):
    """Abstract base class for chunkwise decode inferencers.

    ChunkwiseDecodeInferencers:
    - implements `Inferencer` Protocol.
    - decode new tokens in chunks to reduce device synchronization overhead.

    ChunkwiseDecodeInferencers expect the inheriting class to implement the following abstract methods:
    - `_forward`: Forward the model with query token ids and return the computed logits.
    - `_eos_token_id`: Return the EOS token id according to the model configuration.

    Attributes:
        decode_chunk_size (int): EOS token check interval during decoding, default to 8.
            Used as a simple trick to reduce device synchronization overhead.
    """

    decode_chunk_size: int

    def __init__(self, decode_chunk_size: int = 8) -> None:
        super().__init__()
        self.decode_chunk_size = decode_chunk_size

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

        Stop when `max_new_tokens` is reached or an EOS token is generated.

        Return `DecodeOutput`, which includes the newly generated token ids
        and the logits corresponding to the newly generated tokens.

        If `max_new_tokens <= 0`, return DecodeOutput with empty tensors for both fields.

        Args:
            query_token_ids (torch.Tensor): Query token ids of shape `[batch_size, 1]`
            kv_cache (KVCache): Key-Value cache corresponding to past tokens.
            max_new_tokens (int): Limit on the number of new tokens to generate.
            sample_strategy (SampleStrategy): Token sampling strategy during decoding.

        Returns:
            DecodeOutput: Contains generated new token ids of shape
                `[batch_size, num_generated_tokens]` and token logits of shape
                `[batch_size, num_generated_tokens, vocab_size]`.
                If no new tokens are generated, both fields will be empty tensors.

        Raises:
            ValueError: If `sample_strategy` is unknown.
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
