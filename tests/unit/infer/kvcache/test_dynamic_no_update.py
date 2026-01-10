from typing import TYPE_CHECKING

import pytest

from naive_speculate.infer.kvcache.dynamic import DynamicCache
from naive_speculate.infer.kvcache.dynamic_no_update import DynamicNoUpdateCache
from naive_speculate.testing.infer.kvcache.constants import CROP_RATIOS
from naive_speculate.testing.infer.kvcache.contract import KVCacheContractTests
from naive_speculate.testing.infer.kvcache.utils import get_num_tokens_crop

if TYPE_CHECKING:
    from naive_speculate.infer import KVState


class TestDynamicNoUpdateCacheContract(KVCacheContractTests):
    @pytest.fixture
    def dynamic_no_update_cache(self) -> DynamicNoUpdateCache:
        return DynamicNoUpdateCache()

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

    @pytest.mark.parametrize("crop_ratio", CROP_RATIOS)
    def test_dynamic_cache_crop(
        self,
        dynamic_no_update_cache: DynamicNoUpdateCache,
        kv_states: list[KVState],
        crop_ratio: float,
    ) -> None:
        # bypass the no-update behavior
        DynamicCache.update(dynamic_no_update_cache, kv_states)

        num_tokens = dynamic_no_update_cache.get_num_tokens()
        num_tokens_crop = get_num_tokens_crop(num_tokens, crop_ratio)

        super().crop_test(dynamic_no_update_cache, num_tokens_crop)
