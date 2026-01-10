"""Define `LanguageModel` protocol."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from naive_speculate.infer import KVCache

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
        """Execute forward computation with given token ids and kv cache.

        KV cache will be updated internally with the newly computed key and value tensors.

        Args:
            query_token_ids (torch.Tensor): Input token ids of shape `[batch_size, num_query_tokens]`.
            kv_cache (KVCache): Key value tensors of all past tokens.

        Return the logits at every query token positions, where position `i` gives the logits
        for sampling the token at position `i+1`.
        The shape of output logits is `[batch_size, num_query_tokens, vocab_size]`.

        Returns:
            torch.Tensor: Logits of shape `[batch_size, num_query_tokens, vocab_size]`.
        """
        ...
