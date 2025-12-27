from typing import Protocol

import torch

from .types import DecodeOutput, PrefillOutput, SampleStrategy


class Inferencer(Protocol):
    def prefill(
        self, query_token_ids: torch.Tensor, sample_strategy: SampleStrategy
    ) -> PrefillOutput:
        """Process the `query_token_ids` in parallel and generate the next new tokens.

        KV cache is maintained internally, so expecting `query_token_ids` to
        contain only the new query tokens since the last call to `prefill` or `decode`.

        Return `PrefillOutput`, which includes the generated new token ids and
        the token logits corresponding to the tokens in `query_token_ids`
        (except for the very first tokens) and the newly generated tokens.

        Args:
            input_ids (torch.Tensor): Input token ids of shape `[batch_size, num_query_tokens]`.
            sample_startegy (SampleStartegy): Token sampling strategy for generating new tokens.

        Returns:
            PrefillOutput: Contains generated new token ids of shape `[batch_size, 1]`
                and token logits of shape `[batch_size, num_query_tokens, vocab_size]

        Raises:
            ValueError: If `sample_startegy` is unknown.
        """
        ...

    def decode(
        self,
        query_token_ids: torch.Tensor,
        max_new_tokens: int,
        sample_strategy: SampleStrategy,
    ) -> DecodeOutput:
        """Process `query_token_ids` and auto-regressively generate next new tokens.

        KV cache is maintained internally, so expecting `query_token_ids` to
        contain only the new query tokens since the last call to `prefill` or `decode`.

        Stop when `max_new_tokens` is reached or an EOS token is generated.

        Return `DecodeOutput`, which includes the newly generated token ids
        and the logits corresponding to the newly generated tokens.

        If `max_new_tokens <= 0`, no new tokens will be generated, and
        both `DecodeOutput` fields will be empty tensors.

        Args:
            query_token_ids (torch.Tensor): Input token ids of shape `[batch_size, query_tokens_num]`
            max_new_tokens (int): Limit on the number of new tokens to generate.
            sample_strategy (SampleStrategy): Token sampling strategy during decoding.

        Returns:
            DecodeOutput: Contains generated new token ids of shape
                `[batch_size, num_generated_tokens]` and token logits of shape
                `[batch_size, num_generated_tokens, vocab_size]`.

        Raises:
            ValueError: If `sample_strategy` is unknown.
        """
        ...
