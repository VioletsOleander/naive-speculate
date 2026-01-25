from typing import TYPE_CHECKING

import pytest

from naive_speculate.infer.inferencer.chunkwise import ChunkwiseDecodeInferencer
from naive_speculate.testing.infer.inferencer.constants import MAX_NEW_TOKENS
from naive_speculate.testing.infer.inferencer.contract import InferencerContractTests
from naive_speculate.testing.infer.inferencer.scenario import InferenceScenarioFixtures

if TYPE_CHECKING:
    from naive_speculate.config.strategy import SampleStrategy
    from naive_speculate.testing.infer.inferencer.scenario import InferenceScenario


@pytest.fixture(params=[1, 2, 4, 8])
def decode_chunk_size(request: pytest.FixtureRequest) -> int:
    return request.param


class TestChunkwiseDecodeInferencer(InferencerContractTests, InferenceScenarioFixtures):
    def test_prefill(self, scenario: InferenceScenario, sample_strategy: SampleStrategy) -> None:
        chunkwise_inferencer = ChunkwiseDecodeInferencer(language_model=scenario.fake_lm)
        super().prefill_test(
            inferencer=chunkwise_inferencer,
            query_token_ids=scenario.query_token_ids,
            kv_cache=scenario.kv_cache,
            sample_strategy=sample_strategy,
        )

    @pytest.mark.parametrize("max_new_tokens", MAX_NEW_TOKENS)
    def test_decode(
        self,
        scenario: InferenceScenario,
        max_new_tokens: int,
        sample_strategy: SampleStrategy,
        decode_chunk_size: int,
    ) -> None:
        chunkwise_inferencer = ChunkwiseDecodeInferencer(
            language_model=scenario.fake_lm, decode_chunk_size=decode_chunk_size
        )
        super().decode_test(
            inferencer=chunkwise_inferencer,
            query_token_ids=scenario.query_token_ids,
            kv_cache=scenario.kv_cache,
            max_new_tokens=max_new_tokens,
            sample_strategy=sample_strategy,
        )
