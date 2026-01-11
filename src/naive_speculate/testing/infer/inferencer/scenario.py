"""Define `InferenceScenario` and `InferenceScenarioFixtures` for environment setup of inference tests."""

from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch

from naive_speculate.testing.infer.kvcache.fake import FakeKVCache
from naive_speculate.testing.infer.lm.fake import FakeLanguageModel

from .constants import EOS_POSITIONS, FAKE_LM_CONFIGS, QUERY_SHAPES

if TYPE_CHECKING:
    from naive_speculate.testing.infer.lm.fake import FakeLMConfig

    from .constants import QueryShape

__all__ = ["InferenceScenario", "InferenceScenarioFixtures"]


class InferenceScenario(NamedTuple):
    query_token_ids: torch.Tensor
    fake_lm: FakeLanguageModel
    kv_cache: FakeKVCache


class InferenceScenarioFixtures:
    @pytest.fixture(params=QUERY_SHAPES)
    def query_shape(self, request: pytest.FixtureRequest) -> QueryShape:
        return request.param

    @pytest.fixture(params=FAKE_LM_CONFIGS)
    def lm_config(self, request: pytest.FixtureRequest) -> FakeLMConfig:
        return request.param

    @pytest.fixture(params=EOS_POSITIONS)
    def eos_position(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def scenario(
        self, query_shape: QueryShape, lm_config: FakeLMConfig, eos_position: int
    ) -> InferenceScenario:
        query_token_ids = torch.ones(query_shape, dtype=torch.long)
        fake_lm = FakeLanguageModel(config=lm_config, eos_position=eos_position)
        kv_cache = FakeKVCache()

        return InferenceScenario(
            query_token_ids=query_token_ids, fake_lm=fake_lm, kv_cache=kv_cache
        )
