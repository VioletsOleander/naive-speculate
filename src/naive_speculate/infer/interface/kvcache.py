"""Define `KVCache` and `KVState`."""

from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

__all__ = ["KVCache", "KVState"]


class KVState(NamedTuple):
    """Keys and values tensor for a single transformer layer.

    The shape of `keys` and `values` are
    `[batch_size, num_attention_heads, num_tokens, head_dim]`.

    Attributes:
        keys (torch.Tensor): The keys tensor.
        values (torch.Tensor): The values tensor.
    """

    keys: torch.Tensor
    values: torch.Tensor


class KVCache(Protocol):
    """Stores layerwise key and value tensors, which are used by an Inferencer during inference."""

    def update(self, kv_states: Sequence[KVState]) -> None:
        """Update the storage with new key and value tensors.

        The length of `kv_states` should be equal to the number of transformer layers,
        and each `KVState` contains the new key and value tensors for the corresponding layer.

        All key and value tensors in `kv_states` should have the same shape.

        Args:
            kv_states (Sequence[KVState]): New key and value tensors for each transformer layer.
        """
        ...

    def crop(self, num_tokens_crop: int) -> None:
        """Crop the latest `num_tokens_crop` tokens from the cache.

        `num_tokens_crop` should be non-negative and not exceed the current number of tokens in the cache.

        Args:
            num_tokens_crop (int): Number of latest tokens to crop from the cache.
        """
        ...

    def get_num_tokens(self) -> int:
        """Get the current number of tokens stored in the cache.

        Returns:
            int: Current number of tokens in the cache.
        """
        ...
