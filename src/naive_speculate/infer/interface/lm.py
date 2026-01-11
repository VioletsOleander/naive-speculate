"""Define `LanguageModel`."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from .kvcache import KVCache

__all__ = ["LanguageModel"]


class LanguageModel(Protocol):
    """LanguageModel is able to execute forward computation given input token ids and kv cache.

    LanguageModel possesses pre-trained weights and model configuration such as special token ids,
    vocabulary size, etc.
    """

    @property
    def eos_token_id(self) -> int:
        """Id of the end-of-sequence (EOS) token specified in the model's configuration.

        Eos token id is used to check for the generation stopping criteria.

        Raises:
            ValueError: If the model configuration does not have an eos_token_id.
        """
        ...

    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        """Forward the `query_token_ids` with given `kv_cache`.

        Expect `kv_cache` to contain the key and value tensors for all tokens preceding
        the query tokens.

        `kv_cache` will be updated internally with the newly computed key and value tensors,
        i.e. the key and value tensors corresponding to the query tokens.
        (Currently, I think it simplifies the implementation, but also makes this invocation
        not purely functional, further consideration may be needed in the future.)

        Return the logits at every query token positions, where position `i` gives the logits
        for sampling the token at position `i+1`.
        The shape of output logits is `[batch_size, num_query_tokens, vocab_size]`.

        Args:
            query_token_ids (torch.Tensor): Query token ids of shape `[batch_size, num_query_tokens]`.
            kv_cache (KVCache): Contains the key value tensors of past tokens.

        Returns:
            torch.Tensor: Logits of shape `[batch_size, num_query_tokens, vocab_size]`.
        """
        ...
