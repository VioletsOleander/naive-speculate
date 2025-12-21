import pytest
from transformers import Qwen3ForCausalLM

from naive_speculate.draft import Drafter
from naive_speculate.models import QwenModel
from naive_speculate.utils import SpeculateConfig
from naive_speculate.verify import Verifier


@pytest.fixture(scope="session")
def hf_model(request: pytest.FixtureRequest) -> Qwen3ForCausalLM:
    # request.param should be the model name
    assert isinstance(request.param, str)
    return Qwen3ForCausalLM.from_pretrained(
        request.param, local_files_only=True, device_map="auto", dtype="auto"
    )


@pytest.fixture(scope="session")
def custom_model(request: pytest.FixtureRequest) -> QwenModel:
    # request.param should be the model name
    assert isinstance(request.param, str)
    return QwenModel(request.param)


@pytest.fixture(scope="session")
def drafter(request: pytest.FixtureRequest) -> Drafter:
    # request.param should be the config dict
    assert isinstance(request.param, dict)
    config = SpeculateConfig.from_dict(request.param)
    return Drafter(config)


@pytest.fixture(scope="session")
def verifier(request: pytest.FixtureRequest) -> Verifier:
    # request.param should be the config dict
    assert isinstance(request.param, dict)
    config = SpeculateConfig.from_dict(request.param)
    return Verifier(config)
