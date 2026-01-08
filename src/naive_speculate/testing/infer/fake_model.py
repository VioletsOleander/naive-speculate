from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from naive_speculate.infer import KVCache, KVState


class FakeModelConfig(NamedTuple):
    """Configuration parameters for `FakeModel`.

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


class ForwardResultInjection(NamedTuple):
    """Pre-configured computation results for `FakeModel.forward`.

    Attributes:
        logits (torch.Tensor): The logits to let `forward` generate.
        kv_states (Sequence[KVState]): The key-value states to let `forward` generate.
    """

    logits: torch.Tensor
    kv_states: Sequence[KVState]


class FakeModel:
    """A fake transformer model with causal lm head, implementing `eos_token_id` and `forward`.

    Attributes:
        config (FakeModelConfig): Configuration parameters.
        forward_result (ForwardResultInjection): Injected pre-configured computation results.
    """

    config: FakeModelConfig
    forward_result: ForwardResultInjection | None

    def __init__(
        self,
        config: FakeModelConfig,
    ) -> None:
        self.config = config
        self.forward_result = None

    @property
    def eos_token_id(self) -> int:
        return self.config.eos_token_id

    def forward(self, _query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        """Do a fake forward pass.

        Return the pre-configured logits passed to `__init__` as the token logits,
        and update the `kv_cache` with pre-configured keys and values passed to `__init__`.

        Args:
            _query_token_ids (torch.Tensor): Token ids of shape `[batch_size, num_query_tokens]`.
            kv_cache (KVCache): KVCache to be updated.

        Returns:
            torch.Tensor: The pre-configured logits to return.

        Raises:
            RuntimeError: If `forward_result` is not injected before calling `forward`.
        """
        if self.forward_result is None:
            raise RuntimeError("forward_result is not injected.")

        kv_cache.update(self.forward_result.kv_states)
        return self.forward_result.logits

    def inject_forward_result(self, forward_result: ForwardResultInjection) -> None:
        """Inject pre-configured computation results for `forward`.

        Args:
            forward_result (ForwardResultInjection): Pre-configured computation results.
        """
        self.forward_result = forward_result
