from typing import TYPE_CHECKING

import pytest
import torch

from naive_speculate.config.strategy import SampleStrategy

if TYPE_CHECKING:
    from naive_speculate.infer import Inferencer, KVCache

__all__ = ["InferencerContractTests"]


class InferencerContractTests:
    """Contract tests for `Inferencer` implementations.

    Provide test utility methods corresponding to the methods of `Inferencer`
    and a fixture for `SampleStrategy`.

    `Inferencer` implementations should utilize utility methods defined
    here to test whether they adhere to the expected behavior contracts.
    """

    @pytest.fixture(params=list(SampleStrategy))
    def sample_strategy(self, request: pytest.FixtureRequest) -> SampleStrategy:
        return request.param

    def prefill_test(
        self,
        inferencer: Inferencer,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
        sample_strategy: SampleStrategy,
    ) -> None:
        num_query_tokens = query_token_ids.size(1)
        cache_length_before = kv_cache.get_num_tokens()

        prefill_out = inferencer.prefill(
            query_token_ids=query_token_ids, kv_cache=kv_cache, sample_strategy=sample_strategy
        )
        cache_length_after = kv_cache.get_num_tokens()

        assert prefill_out.token_logits.size()[:-1] == query_token_ids.size()
        assert cache_length_after - cache_length_before == num_query_tokens

        if sample_strategy == SampleStrategy.GREEDY:
            greedy_token_ids = torch.argmax(prefill_out.token_logits[:, -1:], dim=-1)
            assert torch.equal(greedy_token_ids, prefill_out.token_ids)

    def decode_test(
        self,
        inferencer: Inferencer,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
        max_new_tokens: int,
        sample_strategy: SampleStrategy,
    ) -> None:
        prefill_out = inferencer.prefill(
            query_token_ids=query_token_ids, kv_cache=kv_cache, sample_strategy=sample_strategy
        )
        query_token_ids = prefill_out.token_ids

        cache_length_before = kv_cache.get_num_tokens()
        decode_out = inferencer.decode(
            query_token_ids=query_token_ids,
            kv_cache=kv_cache,
            max_new_tokens=max_new_tokens,
            sample_strategy=sample_strategy,
        )
        cache_length_after = kv_cache.get_num_tokens()

        num_generated_tokens = decode_out.token_ids.size(1)

        assert decode_out.token_logits.size(0) == query_token_ids.size(0)
        assert num_generated_tokens <= max_new_tokens
        assert cache_length_after - cache_length_before == num_generated_tokens

        if sample_strategy == SampleStrategy.GREEDY:
            greedy_token_ids = torch.argmax(decode_out.token_logits, dim=-1)
            assert torch.equal(greedy_token_ids, decode_out.token_ids)
