from typing import TYPE_CHECKING

from transformers import DynamicCache, PretrainedConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class DynamicNoUpdateCache:
    """DynamicNoUpdateCache wraps huggingface's DynamicCache, do nothing on update.

    DynamicNoUpdateCache implements KVCache protocol.

    Because huggingface's model implementation will update the passed cache
    during forward as a side effect, therefore this wrapper provides no-op update method.

    The crop method is normally implemented as speculative decoding requires cropping the
    drafter's kvcache.
    """

    cache: DynamicCache

    def __init__(self, model_config: PretrainedConfig) -> None:
        self.cache = DynamicCache(config=model_config)

    def update(self, _keys: Sequence[torch.Tensor], _values: Sequence[torch.Tensor]) -> None:
        pass

    def crop(self, num_tokens_crop: int) -> None:
        raise NotImplementedError("TODO")
