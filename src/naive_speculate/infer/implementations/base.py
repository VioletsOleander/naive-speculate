from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from naive_speculate.infer import DecodeOutput, PrefillOutput, SampleStrategy

from .utils.collection import OutputCollection
from .utils.sample import sample_tokens

if TYPE_CHECKING:
    from collections.abc import Generator


class BaseInferencer(ABC):
    """Abstract base class for inferencers.

    BaseInferencer implements `Inferencer` Protocol by providing simple implementations
    for `prefill` and `decode` methods.

    BaseInferencer expects the inheriting class to implement the following abstract methods:
    - `_forward`: Forward with query token ids and return the computed logits.
    - `_eos_token_id`: Return the EOS token id.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def _eos_token_id(self) -> int:
        """The EOS token id."""
        ...

    @abstractmethod
    def _forward(self, query_token_ids: torch.Tensor) -> torch.Tensor:
        """Forward with `query_token_ids` and return the computed logits.

        KV cache is expected to be managed by this method's implementation internally.

        Args:
            query_token_ids (torch.Tensor): Query token ids of shape `[batch_size, num_query_tokens]`.

        Returns:
            torch.Tensor: Computed logits of shape `[batch_size, num_query_tokens, vocab_size]`.
        """
        ...

    def _generation_stream(
        self,
        query_token_ids: torch.Tensor,
        sample_strategy: SampleStrategy,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
        """Generate new tokens auto-regressively as a stream.

        Each iteration performs:
        1. Forward the model with `query_token_ids` to get logits.
        2. Sample the next token ids from the logits according to `sample_strategy`.
        3. Update `query_token_ids` with the newly sampled token ids.

        Args:
            query_token_ids (torch.Tensor): Initial query token ids of shape `[batch_size, num_query_tokens]`.
            sample_strategy (SampleStrategy): Sampling strategy for new token generation.

        Yields:
            tuple[torch.Tensor, torch.Tensor]: Sampled new token ids of shape `[batch_size, 1]`,
                and computed logits of shape `[batch_size, num_query_tokens, vocab_size]` (the first call)
                or of shape `[batch_size, 1, vocab_size]` (subsequent calls).
        """
        while True:
            # 1. Forward
            logits = self._forward(query_token_ids)

            # 2. Sample
            next_token_logits = logits[:, -1].to(dtype=torch.float32, copy=True)
            next_token_ids = sample_tokens(next_token_logits, sample_strategy)

            # 3. Update
            query_token_ids = next_token_ids

            yield next_token_ids, logits

    @torch.no_grad()
    def prefill(
        self, query_token_ids: torch.Tensor, sample_strategy: SampleStrategy
    ) -> PrefillOutput:
        """Process the `query_token_ids` in parallel and generate the next token.

        Return the generated new token ids and logits corresponding to the `query_token_ids`
        (except for the first tokens) and the newly generated token.

        Args:
            query_token_ids (torch.Tensor): Query token ids of shape `[batch_size, num_query_tokens]`.
            sample_strategy (SampleStrategy): Token sampling strategy during prefill.

        Returns:
            PrefillOutput: Collection of output token ids of shape
                `[batch_size, 1]` and logits of shape `[batch_size, num_query_tokens, vocab_size]`.

        Raises:
            ValueError: If `sample_strategy` is unknown.
        """
        output_collection = OutputCollection()

        stream = self._generation_stream(
            query_token_ids=query_token_ids, sample_strategy=sample_strategy
        )
        output_ids, output_logits = next(stream)
        output_collection.update(output_ids, output_logits)

        return PrefillOutput._make(output_collection.finalize())

    @torch.no_grad()
    def decode(
        self,
        query_token_ids: torch.Tensor,
        max_new_tokens: int,
        sample_strategy: SampleStrategy,
    ) -> DecodeOutput:
        """Process `query_token_ids` and auto-regressively generate next new tokens.

        Check for EOS token after each generation iteration, which means device
        synchronization will happen at each iteration.

        Stop when `max_new_tokens` is reached or an EOS token is generated.

        Return `DecodeOutput`, which includes the newly generated token ids
        and the logits corresponding to the newly generated tokens.

        If `max_new_tokens <= 0`, return DecodeOutput with empty tensors for both fields.

        Args:
            query_token_ids (torch.Tensor): Query token ids of shape `[batch_size, 1]`
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
            query_token_ids=query_token_ids, sample_strategy=sample_strategy
        )
        for _ in range(max_new_tokens):
            output_ids, output_logits = next(stream)
            output_collection.update(output_ids, output_logits)

            # Check for EOS token in the newly generated tokens
            # currently only reasonable for batch_size=1
            if (output_ids == self._eos_token_id).any().item():
                break

        return DecodeOutput._make(output_collection.finalize())
