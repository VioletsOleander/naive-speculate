from typing import TYPE_CHECKING, override

from transformers import DynamicCache

from naive_speculate.infer import KVCache

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class DynamicNoUpdateCache(KVCache):
    """DynamicNoUpdateCache wraps huggingface's DynamicCache, do nothing on update.

    DynamicNoUpdateCache implements KVCache protocol.

    Because huggingface's model implementation will update the passed cache
    during forward as a side effect, therefore this wrapper provides no-op update method.

    The crop method is normally implemented as speculative decoding requires cropping the
    drafter's kvcache.
    """

    cache: DynamicCache

    def __init__(self) -> None:
        self.cache = DynamicCache()

    @override
    def update(self, keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor]) -> None:
        """Intentionally a no-op, the underlying model will update `self.cache` in-place."""

    @override
    def crop(self, num_tokens_crop: int) -> None:
        length = self.cache.get_seq_length(0)
        self.cache.crop(length - num_tokens_crop)
