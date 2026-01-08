from abc import abstractmethod
from typing import TYPE_CHECKING, override

import torch

from naive_speculate.infer import DecodeOutput, KVCache, PrefillOutput
from naive_speculate.infer import Inferencer as InferencerProtocol
from naive_speculate.utils.sample import sample_tokens

from .utils.collection import OutputCollection

if TYPE_CHECKING:
    from collections.abc import Generator

    from naive_speculate.config.strategy import SampleStrategy


class BasicInferencer(InferencerProtocol):
    """Basic Inferencer implements `prefill` and `decode` method of the `Inferencer` protocol.

    BaseInferencer utilize `forward` method to provide simple implementations
    for `prefill` and `decode` methods, and leave the concrete implementation
    of `forward` to inheriting concrete classes.

    BaseInferencer expects inheriting classes to implement the following abstract methods:
    - `forward`: Forward with query token ids and return the computed logits.
    - `_eos_token_id`: Return the EOS token id.
    """

    @property
    @abstractmethod
    def _eos_token_id(self) -> int:
        """Id of the end-of-sequence (EOS) token."""
        ...

    @abstractmethod
    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor: ...

    @torch.no_grad()
    @override
    def prefill(
        self, query_token_ids: torch.Tensor, kv_cache: KVCache, sample_strategy: SampleStrategy
    ) -> PrefillOutput:
        output_collection = OutputCollection()

        stream = self._generation_stream(
            query_token_ids=query_token_ids, kv_cache=kv_cache, sample_strategy=sample_strategy
        )
        output_ids, output_logits = next(stream)
        output_collection.update(output_ids, output_logits)

        return PrefillOutput._make(output_collection.finalize())

    @torch.no_grad()
    @override
    def decode(
        self,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
        max_new_tokens: int,
        sample_strategy: SampleStrategy,
    ) -> DecodeOutput:
        """Process `query_token_ids` and auto-regressively generate next new tokens.

        Check for EOS token after each generation iteration, which means device
        synchronization will happen at each iteration.

        Refers to the interface `Inferencer.decode` for more details.
        """
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

        output_collection = OutputCollection()

        stream = self._generation_stream(
            query_token_ids=query_token_ids, kv_cache=kv_cache, sample_strategy=sample_strategy
        )
        for _ in range(max_new_tokens):
            output_ids, output_logits = next(stream)
            output_collection.update(output_ids, output_logits)

            # Check for EOS token in the newly generated tokens
            # currently only reasonable for batch_size=1
            if (output_ids == self._eos_token_id).any().item():
                break

        return DecodeOutput._make(output_collection.finalize())

    def _generation_stream(
        self,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
        sample_strategy: SampleStrategy,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
        """Generate new tokens auto-regressively as a stream.

        Each iteration performs:
        1. Forward the model with `query_token_ids` to get logits.
        2. Sample the next token ids from the logits according to `sample_strategy`.
        3. Update `query_token_ids` with the newly sampled token ids.

        Args:
            query_token_ids (torch.Tensor): Initial query token ids of shape `[batch_size, num_query_tokens]`.
            kv_cache (KVCache): Keys and values tensors corresponding to the past tokens.
            sample_strategy (SampleStrategy): Sampling strategy for new token generation.

        Yields:
            tuple[torch.Tensor, torch.Tensor]: Sampled new token ids of shape `[batch_size, 1]`,
                and computed logits.
                The computed logits are of shape `[batch_size, num_query_tokens, vocab_size]`
                for the first call, and of shape `[batch_size, 1, vocab_size]` for subsequent calls.
        """
        while True:
            # 1. Forward
            token_logits = self.forward(query_token_ids, kv_cache)

            # 2. Sample
            next_token_logits = token_logits[:, -1].to(dtype=torch.float32, copy=True)
            next_token_ids = sample_tokens(next_token_logits, sample_strategy)

            # 3. Update
            query_token_ids = next_token_ids

            yield next_token_ids, token_logits
