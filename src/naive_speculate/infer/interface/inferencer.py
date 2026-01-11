"""Define `Inferencer`, `PrefillOutput` and `DecodeOutput`."""

from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    import torch

    from naive_speculate.config.strategy import SampleStrategy

    from .kvcache import KVCache

__all__ = ["DecodeOutput", "Inferencer", "PrefillOutput"]


class PrefillOutput(NamedTuple):
    """Output of `Inferencer.prefill` method.

    Attributes:
        token_ids (torch.Tensor): The newly generated token ids after prefill. Shape `[batch_size, 1]`.
        token_logits (torch.Tensor): The logits at the query token positions.
            Shape `[batch_size, num_query_tokens, vocab_size]`.
    """

    token_ids: torch.Tensor
    token_logits: torch.Tensor


class DecodeOutput(NamedTuple):
    """Output of `Inferencer.decode` method.

    Attributes:
        token_ids (torch.Tensor): The newly generated token ids after decode.
            Shape `[batch_size, num_generated_tokens]`.
        token_logits (torch.Tensor): The logits used to sample the newly generated tokens.
            Shape `[batch_size, num_generated_tokens, vocab_size]`.
    """

    token_ids: torch.Tensor
    token_logits: torch.Tensor


class Inferencer(Protocol):
    """Inferencer is able to process token sequences and generate new tokens.

    Inferencer processes token sequences and generates new tokens using specified sampling strategies.

    Inferencer should either be itself a transformer model or be able to delegate the
    token processing to a transformer model. The transformer model is expected to support using
    KV cache to avoid redundant computations during inference.

    Inferencer will update the KVCache internally when processing the query tokens.
    """

    def prefill(
        self, query_token_ids: torch.Tensor, kv_cache: KVCache, sample_strategy: SampleStrategy
    ) -> PrefillOutput:
        """Process the `query_token_ids` in parallel and generate the next new tokens.

        Expect `kv_cache` to contain the key and value tensors for all tokens preceding
        the query tokens.

        `kv_cache` will be updated internally with the newly computed key and value tensors,
        i.e. the key and value tensors corresponding to the query tokens.

        Return `PrefillOutput`, which includes:
        - the generated new token ids. Shape `[batch_size, 1]`.
        - the token logits at the query token positions.
            Shape `[batch_size, num_query_tokens, vocab_size]`.

        Args:
            query_token_ids (torch.Tensor): Query token ids of shape `[batch_size, num_query_tokens]`.
            kv_cache (KVCache): Contains the past key and value tensors for each transformer layer.
            sample_strategy (SampleStrategy): Token sampling strategy for generating new tokens.

        Returns:
            PrefillOutput: Contains generated new token ids of shape `[batch_size, 1]`
                and token logits of shape `[batch_size, num_query_tokens, vocab_size]`.
        """
        ...

    def decode(
        self,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
        max_new_tokens: int,
        sample_strategy: SampleStrategy,
    ) -> DecodeOutput:
        """Process `query_token_ids` and auto-regressively generate next new tokens.

        Expect `kv_cache` to contain the key and value tensors for all tokens preceding
        the query tokens.

        `kv_cache` will be updated internally with the newly computed key and value tensors,
        i.e. the key and value tensors corresponding to the query tokens.

        Expect `query_token_ids` to contain only the new query tokens
        since the last call to `prefill` or `decode`, i.e., of shape `[batch_size, 1]`.

        Stop when `max_new_tokens` is reached or an EOS token is generated.

        Return `DecodeOutput`, which includes:
        - the newly generated token ids. Shape `[batch_size, num_generated_tokens]`.
        - the logits used to sample the newly generated tokens.
            Shape `[batch_size, num_generated_tokens, vocab_size]`.

        Args:
            query_token_ids (torch.Tensor): Query token ids of shape `[batch_size, 1]`
            kv_cache (KVCache): Contains the past key and value tensors for each transformer layer.
            max_new_tokens (int): Limit on the number of new tokens to generate, should be positive (`> 0`).
            sample_strategy (SampleStrategy): Token sampling strategy during decoding.

        Returns:
            DecodeOutput: Contains generated new token ids of shape
                `[batch_size, num_generated_tokens]` and token logits of shape
                `[batch_size, num_generated_tokens, vocab_size]`.
        """
        ...
