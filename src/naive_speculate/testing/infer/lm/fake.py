"""Provide `FakeLanguageModel` and its configuration `FakeLMConfig`."""

from typing import TYPE_CHECKING, NamedTuple, override

import torch

from naive_speculate.infer import KVState, LanguageModel

if TYPE_CHECKING:
    from naive_speculate.infer import KVCache

__all__ = ["FakeLMConfig", "FakeLanguageModel"]


class FakeLMConfig(NamedTuple):
    """Configuration parameters for `FakeLanguageModel`.

    Attributes:
        eos_token_id (int): End-of-sequence token id.
        vocab_size (int): Number of tokens in the vocabulary.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads for multi-head attention.
        embed_dim (int): Dimension of the token embeddings.
    """

    eos_token_id: int
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int


class FakeLanguageModel(LanguageModel):
    """Lightweight fake implementation of `LanguageModel`.

    Attributes:
        config (FakeLMConfig): Configuration parameters.
        eos_position (int): Position at which the model will start predicting `eos_token_id`.
    """

    config: FakeLMConfig
    eos_position: int
    _forward_count: int

    def __init__(self, config: FakeLMConfig, eos_position: int) -> None:
        self.config = config
        self.eos_position = eos_position
        self._forward_count = 0

    @property
    @override
    def eos_token_id(self) -> int:
        return self.config.eos_token_id

    @override
    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        """Perform a fake forward pass.

        The output logits are randomly generated.
        The kv states to be stored in the kv cache are empty tensors with appropriate shapes.

        `forward` maintains an internal counter, if the number of invocations reaches `eos_position`,
        the output logits will be set to always predict the `eos_token_id`.
        """
        batch_size, num_query_tokens = query_token_ids.size()
        logits = torch.randn(batch_size, num_query_tokens, self.config.vocab_size)

        kv_shape = (
            batch_size,
            self.config.num_heads,
            num_query_tokens,
            self.config.embed_dim // self.config.num_heads,
        )
        kv_states = [
            KVState(keys=torch.empty(kv_shape), values=torch.empty(kv_shape))
            for _ in range(self.config.num_layers)
        ]
        kv_cache.update(kv_states)

        self._forward_count += 1
        if self._forward_count >= self.eos_position:
            logits[:, -1, self.config.eos_token_id] = torch.finfo(logits.dtype).max

        return logits
