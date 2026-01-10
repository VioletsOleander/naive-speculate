from typing import TYPE_CHECKING

import pytest

from naive_speculate.infer.kvcache.dynamic_cache import DynamicCache
from naive_speculate.testing.infer.kvcache.constants import CROP_RATIOS
from naive_speculate.testing.infer.kvcache.contract import KVCacheContractTests
from naive_speculate.testing.infer.kvcache.utils import get_num_tokens_crop

if TYPE_CHECKING:
    from naive_speculate.infer import KVState


class TestDynamicCacheContract(KVCacheContractTests):
    @pytest.fixture
    def dynamic_cache(self) -> DynamicCache:
        return DynamicCache()

    def test_dynamic_cache_update(
        self, dynamic_cache: DynamicCache, kv_states: list[KVState]
    ) -> None:
        super().update_test(dynamic_cache, kv_states)

    @pytest.mark.parametrize("crop_ratio", CROP_RATIOS)
    def test_dynamic_cache_crop(
        self, dynamic_cache: DynamicCache, kv_states: list[KVState], crop_ratio: float
    ) -> None:
        dynamic_cache.update(kv_states)

        num_tokens = dynamic_cache.get_num_tokens()
        num_tokens_crop = get_num_tokens_crop(num_tokens, crop_ratio)

        super().crop_test(dynamic_cache, num_tokens_crop)
