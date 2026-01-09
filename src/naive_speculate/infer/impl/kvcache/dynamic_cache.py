from collections.abc import Sequence
from typing import overload, override

import torch
import transformers

from naive_speculate.infer import KVCache, KVState


class DynamicCache(KVCache, Sequence):
    """DynamicCache wraps huggingface's DynamicCache.

    DynamicCache implements `KVCache` and `Sequence`.

    Attributes:
        cache (transformers.DynamicCache): The underlying huggingface DynamicCache instance.
    """

    cache: transformers.DynamicCache

    def __init__(self) -> None:
        self.cache = transformers.DynamicCache()

    @override
    def update(self, kv_states: Sequence[KVState]) -> None:
        for layer_idx, (k, v) in enumerate(kv_states):
            self.cache.update(key_states=k, value_states=v, layer_idx=layer_idx)

    @override
    def crop(self, num_tokens_crop: int) -> None:
        if num_tokens_crop <= 0:
            return

        length = self.cache.get_seq_length(0)
        if num_tokens_crop > length:
            raise ValueError(
                f"Cannot crop {num_tokens_crop} tokens from cache with length {length}."
            )

        self.cache.crop(length - num_tokens_crop)

    @override
    def get_num_tokens(self) -> int:
        return self.cache.get_seq_length(0)

    @overload
    def __getitem__(self, index: int) -> KVState: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[KVState, ...]: ...

    @override
    def __getitem__(self, index: int | slice) -> KVState | tuple[KVState, ...]:
        """Retrieve kv states from the given layer index or indices.

        If kv states for a certain layer do not exist,
        corresponding keys and values will be empty tensors.

        Args:
            index (int | slice): The layer index or slice to retrieve.

        Returns:
            KVState | tuple[KVState, ...]: The kv state(s) of the specified layer(s).
        """

        def to_kv_state(layer: transformers.cache_utils.CacheLayerMixin) -> KVState:
            keys = layer.keys or torch.empty(0)
            values = layer.values or torch.empty(0)
            return KVState(keys=keys, values=values)

        if isinstance(index, int):
            layer = self.cache.layers[index]
            return to_kv_state(layer)

        if isinstance(index, slice):
            layers = self.cache.layers[index]
            return tuple(map(to_kv_state, layers))

        raise TypeError(f"Invalid index type: {type(index)}")

    @override
    def __len__(self) -> int:
        return len(self.cache.layers)


KVCacheImpl = DynamicCache
