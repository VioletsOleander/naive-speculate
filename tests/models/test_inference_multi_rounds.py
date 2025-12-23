import pytest
import torch
from transformers import Qwen3ForCausalLM

from naive_speculate.models import QwenModel

from .constants import MAX_NEW_TOKENS, MODEL_NAME, PROMPT_LENGTH

MAX_NEW_TOKENS_PER_ROUND = 3


@pytest.mark.parametrize(
    "hf_model, custom_model", [(MODEL_NAME, MODEL_NAME)], indirect=True
)
def test_inference_multi_rounds(
    hf_model: Qwen3ForCausalLM, custom_model: QwenModel
) -> None:
    """Verify that the custom model's outputs match those of the Hugging Face model for multiple rounds inference."""
    input_ids = torch.randint(0, hf_model.config.vocab_size, (1, PROMPT_LENGTH))

    custom_model._reset()
    total_len = PROMPT_LENGTH
    model_inputs = input_ids
    model_outputs = None
    while True:
        if total_len >= PROMPT_LENGTH + MAX_NEW_TOKENS:
            break
        model_outputs = custom_model.inference(
            input_ids=model_inputs,
            max_new_tokens=MAX_NEW_TOKENS_PER_ROUND,
            decode_method="greedy",
        )
        total_len = model_outputs.shape[1]
        model_inputs = model_outputs
    assert model_outputs is not None

    # let hf model not reuse the kv_cache allocation from custom model
    hf_outputs = hf_model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    assert isinstance(hf_outputs, torch.Tensor)
    hf_outputs_length = hf_outputs.shape[1]
    assert torch.equal(hf_outputs, model_outputs[:, :hf_outputs_length])

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
    hf_outputs_length = hf_outputs.shape[1]
    assert torch.equal(hf_outputs, model_outputs[:, :hf_outputs_length])
