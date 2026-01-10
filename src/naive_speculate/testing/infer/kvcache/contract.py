from typing import TYPE_CHECKING

import pytest

from .constants import KVSTATES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from naive_speculate.infer import KVCache, KVState


class KVCacheContractTests:
    """Contract tests for `KVCache` implementations.

    `KVCache` implementations should utilize utility functions defined
    here to test whether they adhere to the expected behavior contracts.
    """

    @pytest.fixture(params=KVSTATES)
    def kv_states(self, request: pytest.FixtureRequest) -> list[KVState]:
        return request.param

    def update_test(self, kv_cache: KVCache, kv_states: Sequence[KVState]) -> None:
        num_tokens_new = kv_states[0].keys.size(-2)

        num_tokens_before = kv_cache.get_num_tokens()
        kv_cache.update(kv_states=kv_states)
        num_tokens_after = kv_cache.get_num_tokens()

        assert num_tokens_after == num_tokens_before + num_tokens_new

    def crop_test(self, kv_cache: KVCache, num_tokens_crop: int) -> None:
        num_tokens_before = kv_cache.get_num_tokens()
        kv_cache.crop(num_tokens_crop=num_tokens_crop)
        num_tokens_after = kv_cache.get_num_tokens()

        assert num_tokens_after == num_tokens_before - num_tokens_crop
