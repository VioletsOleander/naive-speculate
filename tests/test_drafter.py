import pytest
import torch
from transformers import Qwen3ForCausalLM
from transformers.generation.utils import GenerateDecoderOnlyOutput

from naive_speculate.draft import Drafter

PROMPT_LENGTH = 16
DRAFT_MODEL_NAME = "Qwen/Qwen3-0.6B"
CONFIG_DICT = {
    "max_new_tokens": 32768,
    "decode_method": "greedy",
    "drafter_model_name": DRAFT_MODEL_NAME,
    "draft_tokens_num": 10,
    "verifier_model_name": DRAFT_MODEL_NAME,
}


@pytest.mark.parametrize(
    "drafter, hf_model", [(CONFIG_DICT, DRAFT_MODEL_NAME)], indirect=True
)
def test_drafter(drafter: Drafter, hf_model: Qwen3ForCausalLM):
    """Verify drafter model outputs match HuggingFace model outputs."""
    input_ids = torch.randint(0, hf_model.config.vocab_size, (1, PROMPT_LENGTH))

    drafter.kv_cache.crop(0)
    draft_outputs, candidate_logits = drafter.draft(input_ids)

    drafter.kv_cache.crop(0)
    hf_outputs = hf_model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=CONFIG_DICT["draft_tokens_num"],
        do_sample=False,
        use_cache=True,
        past_key_values=drafter.kv_cache,
        output_logits=True,
        logits_to_keep=0,  # keep all logits
        return_dict_in_generate=True,
    )
    assert isinstance(hf_outputs, GenerateDecoderOnlyOutput)
    assert torch.equal(hf_outputs.sequences, draft_outputs)

    assert hf_outputs.logits is not None
    hf_logits = torch.cat(hf_outputs.logits, dim=0).unsqueeze(0)
    assert torch.equal(candidate_logits, hf_logits)
