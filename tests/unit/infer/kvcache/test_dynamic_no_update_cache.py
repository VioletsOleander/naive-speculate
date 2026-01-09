from typing import TYPE_CHECKING

import pytest

from naive_speculate.infer.impl.kvcache.dynamic_no_update_cache import (
    DynamicCache,
    DynamicNoUpdateCache,
)
from naive_speculate.testing.infer.kvcache import KVSTATES, NUM_TOKENS_CROP, KVCacheContractTests

if TYPE_CHECKING:
    from naive_speculate.infer import KVState


class TestDynamicNoUpdateCacheContract(KVCacheContractTests):
    @pytest.fixture
    def dynamic_no_update_cache(self) -> DynamicNoUpdateCache:
        return DynamicNoUpdateCache()

    @pytest.fixture(params=KVSTATES)
    def kv_states(self, request: pytest.FixtureRequest) -> list[KVState]:
        return request.param

    @pytest.mark.xfail(reason="DynamicNoUpdateCache does not support updates", strict=True)
    def test_dynamic_cache_update_xfail(
        self, dynamic_no_update_cache: DynamicNoUpdateCache, kv_states: list[KVState]
    ) -> None:
        super().update_test(dynamic_no_update_cache, kv_states)

    def test_dynamic_cache_update(
        self, dynamic_no_update_cache: DynamicNoUpdateCache, kv_states: list[KVState]
    ) -> None:
        # bypass the no-update behavior
        DynamicCache.update(dynamic_no_update_cache, kv_states)

        num_tokens_before = dynamic_no_update_cache.get_num_tokens()
        dynamic_no_update_cache.update(kv_states)
        num_tokens_after = dynamic_no_update_cache.get_num_tokens()
        assert num_tokens_before == num_tokens_after

    @pytest.mark.parametrize("num_tokens_crop", NUM_TOKENS_CROP)
    def test_dynamic_cache_crop(
        self,
        dynamic_no_update_cache: DynamicNoUpdateCache,
        kv_states: list[KVState],
        num_tokens_crop: int,
    ) -> None:
        # bypass the no-update behavior
        DynamicCache.update(dynamic_no_update_cache, kv_states)
        super().crop_test(dynamic_no_update_cache, num_tokens_crop)
