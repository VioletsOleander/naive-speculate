from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from transformers import DynamicCache


class DynamicNoUpdateCache:
    """DynamicNoUpdateCache warps huggingface's DynamicCache, do nothing on update.

    DynamicNoUpdateCache implements KVCache protocol.

    Because huggingface's model implementation will update the passed cache
    during forward as a side effect, therefore this warpper provides no-op update method.

    The crop method is normally implemented as speculative decoding requires cropping the
    drafter's kvcache.
    """

    cache: DynamicCache

    def update(self, _keys: Sequence[torch.Tensor], _values: Sequence[torch.Tensor]) -> None:
        pass

    def crop(self, num_tokens_crop: int) -> None:
        raise NotImplementedError("TODO")
