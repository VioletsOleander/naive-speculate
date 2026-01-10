from typing import TYPE_CHECKING

from naive_speculate.infer import KVCache, KVState

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["FakeKVCache"]


class FakeKVCache(KVCache):
    """Lightweight fake implementation of `KVCache`.

    Attributes:
        num_tokens (int): The total number of tokens stored in the cache.
    """

    num_tokens: int

    def __init__(self) -> None:
        self.num_tokens = 0

    def update(self, kv_states: Sequence[KVState]) -> None:
        self.num_tokens += kv_states[0].keys.size(-2)

    def crop(self, num_tokens_crop: int) -> None:
        self.num_tokens -= num_tokens_crop

    def get_num_tokens(self) -> int:
        return self.num_tokens
