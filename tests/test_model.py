import pytest
import torch
from transformers import Qwen3ForCausalLM

from naive_speculate.utils import QwenModel

CONFIG_DICT = {
    "model_name": "Qwen/Qwen3-0.6B",
    "max_new_tokens": 10,
    "decode_method": "greedy",
    "prompt_length": 16,
}


@pytest.fixture
def hf_model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM.from_pretrained(
        CONFIG_DICT["model_name"], local_files_only=True
    )


@pytest.fixture
def custom_model() -> QwenModel:
    return QwenModel(CONFIG_DICT["model_name"])


def test_self_consistency(custom_model: QwenModel):
    """Verify that the model produces consistent outputs across multiple runs with the same input."""
    input_ids = torch.randint(
        0, custom_model.model.config.vocab_size, (1, CONFIG_DICT["prompt_length"])
    )

    output1 = custom_model.inference(
        input_ids=input_ids,
        max_new_tokens=CONFIG_DICT["max_new_tokens"],
        decode_method="greedy",
    )

    custom_model.kv_cache.crop(0)
    output2 = custom_model.inference(
        input_ids=input_ids,
        max_new_tokens=CONFIG_DICT["max_new_tokens"],
        decode_method="greedy",
    )

    assert torch.equal(output1, output2)


def test_model(hf_model: Qwen3ForCausalLM, custom_model: QwenModel):
    """Verify that the custom model's outputs match those of the Hugging Face model."""
    input_ids = torch.randint(
        0, hf_model.config.vocab_size, (1, CONFIG_DICT["prompt_length"])
    )

    custom_outputs = custom_model.inference(
        input_ids=input_ids,
        max_new_tokens=CONFIG_DICT["max_new_tokens"],
        decode_method="greedy",
    )

    # weirdly, if we don't let hf model reuse the kv_cache allocation from custom model, the outputs differ
    custom_model.kv_cache.crop(0)
    hf_outputs = hf_model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=CONFIG_DICT["max_new_tokens"],
        do_sample=False,
        use_cache=True,
        past_key_values=custom_model.kv_cache,
    )

    assert isinstance(hf_outputs, torch.Tensor)
    assert torch.equal(hf_outputs, custom_outputs)
