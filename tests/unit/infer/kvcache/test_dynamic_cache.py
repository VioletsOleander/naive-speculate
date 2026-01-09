from typing import TYPE_CHECKING

import pytest

from naive_speculate.infer.impl.kvcache.dynamic_cache import DynamicCache
from naive_speculate.testing.infer.kvcache import KVSTATES, NUM_TOKENS_CROP, KVCacheContractTests

if TYPE_CHECKING:
    from naive_speculate.infer import KVState


class TestDynamicCacheContract(KVCacheContractTests):
    @pytest.fixture
    def dynamic_cache(self) -> DynamicCache:
        return DynamicCache()

    @pytest.fixture(params=KVSTATES)
    def kv_states(self, request: pytest.FixtureRequest) -> list[KVState]:
        return request.param

    def test_dynamic_cache_update(
        self, dynamic_cache: DynamicCache, kv_states: list[KVState]
    ) -> None:
        return super().update_test(dynamic_cache, kv_states)

    @pytest.mark.parametrize("num_tokens_crop", NUM_TOKENS_CROP)
    def test_dynamic_cache_crop(
        self, dynamic_cache: DynamicCache, kv_states: list[KVState], num_tokens_crop: int
    ) -> None:
        dynamic_cache.update(kv_states)
        return super().crop_test(dynamic_cache, num_tokens_crop)
