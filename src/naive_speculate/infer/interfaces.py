from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from naive_speculate.utils.sample import SampleStrategy


class PrefillOutput(NamedTuple):
    """Output of `Inferencer.prefill` method.

    Attributes:
        token_ids (torch.Tensor): The newly generated token ids after prefill. Shape `[batch_size, 1]`.
        token_logits (torch.Tensor): The logits of the query tokens (excluding the first ones)
            and the newly generated tokens. Shape `[batch_size, num_query_tokens, vocab_size]`.
    """

    token_ids: torch.Tensor
    token_logits: torch.Tensor


class DecodeOutput(NamedTuple):
    """Output of `Inferencer.decode` method.

    Attributes:
        token_ids (torch.Tensor): The newly generated token ids after decode.
            Shape `[batch_size, num_generated_tokens]`.
        token_logits (torch.Tensor): The logits of newly generated tokens.
            Shape `[batch_size, num_generated_tokens, vocab_size]`.
    """

    token_ids: torch.Tensor
    token_logits: torch.Tensor


class KVCache(Protocol):
    """Stores layerwise key and value tensors, which are used by an Inferencer during inference."""

    def update(self, keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor]) -> None:
        """Update the storage with new key and value tensors.

        `keys` and `values` are sequences of tensors. The length of the sequences
        should be equal to the number of transformer layers, and each tensor
        in the sequences corresponds to the key or value tensor of a transformer layer.

        Args:
            keys (Sequence[torch.Tensor]): New key tensors for each transformer layer.
            values (Sequence[torch.Tensor]): New value tensors for each transformer layer.
        """
        ...

    def crop(self, num_tokens_crop: int) -> None:
        """Crop the latest `num_tokens_crop` tokens from the cache.

        Args:
            num_tokens_crop (int): Number of latest tokens to crop from the cache.
        """
        ...


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

        `kv_cache` is used for avoiding redundant computations for the key and value tensors.

        `kv_cache` will be updated internally with the newly computed key and value tensors,
        i.e. the key and value tensors corresponding to the query tokens.
        (Currently, I think it simplifies the implementation, but also makes this invocation
        not purely functional, further consideration may be needed in the future.)

        Return `PrefillOutput`, which includes:
        - the generated new token ids. Shape `[batch_size, 1]`.
        - the token logits corresponding to the query tokens
            (except for the first tokens) and the newly generated tokens.
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

        `kv_cache` is used for avoiding redundant computations for the key and value tensors.

        `kv_cache` will be updated internally with the newly computed key and value tensors,
        i.e. the key and value tensors corresponding to the query tokens.
        (Currently, I think it simplifies the implementation, but also makes this invocation
        not purely functional, further consideration may be needed in the future.)

        Expect `query_token_ids` to contain only the new query tokens
        since the last call to `prefill` or `decode`, i.e., of shape `[batch_size, 1]`.

        Stop when `max_new_tokens` is reached or an EOS token is generated.

        Return `DecodeOutput`, which includes:
        - the newly generated token ids. Shape `[batch_size, num_generated_tokens]`.
        - the logits corresponding to the newly generated tokens.
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
