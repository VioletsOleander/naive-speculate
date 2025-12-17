import pytest
import torch
from transformers import Qwen3ForCausalLM

from naive_speculate.models import QwenModel

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 10
PROMPT_LENGTH = 16


@pytest.mark.parametrize("custom_model", [MODEL_NAME], indirect=True)
def test_self_consistency(custom_model: QwenModel) -> None:
    """Verify that the model produces consistent outputs across multiple runs with the same input."""
    input_ids = torch.randint(
        0, custom_model.model.config.vocab_size, (1, PROMPT_LENGTH)
    )

    custom_model.kv_cache.crop(0)
    output1 = custom_model.inference(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        decode_method="greedy",
    )

    custom_model.kv_cache.crop(0)
    output2 = custom_model.inference(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        decode_method="greedy",
    )

    assert torch.equal(output1, output2)


@pytest.mark.parametrize(
    "hf_model, custom_model", [(MODEL_NAME, MODEL_NAME)], indirect=True
)
def test_model(hf_model: Qwen3ForCausalLM, custom_model: QwenModel) -> None:
    """Verify that the custom model's outputs match those of the Hugging Face model."""
    input_ids = torch.randint(0, hf_model.config.vocab_size, (1, PROMPT_LENGTH))

    custom_model.kv_cache.crop(0)
    custom_outputs = custom_model.inference(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        decode_method="greedy",
    )

    # let hf model reuse the kv_cache allocation from custom model
    custom_model.kv_cache.crop(0)
    hf_outputs = hf_model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        past_key_values=custom_model.kv_cache,
    )

    assert isinstance(hf_outputs, torch.Tensor)
    assert torch.equal(hf_outputs, custom_outputs)
