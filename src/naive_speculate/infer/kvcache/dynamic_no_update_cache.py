from typing import TYPE_CHECKING, override

from .dynamic_cache import DynamicCache

if TYPE_CHECKING:
    from collections.abc import Sequence

    from naive_speculate.infer import KVState


class DynamicNoUpdateCache(DynamicCache):
    """DynamicNoUpdateCache do nothing on update.

    Because huggingface's model implementation will update the passed cache
    during forward as a side effect, therefore this wrapper provides no-op update method.

    Refers to the base class `DynamicCache` for more details.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def update(self, kv_states: Sequence[KVState]) -> None:
        """Intentionally a no-op, the underlying model will update `self.cache` in-place."""


KVCacheImpl = DynamicNoUpdateCache
