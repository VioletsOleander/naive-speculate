import pytest
import torch
from transformers import Qwen3ForCausalLM
from transformers.generation.utils import GenerateDecoderOnlyOutput

from naive_speculate.draft import Drafter

from .constants import CONFIG_DICT, DRAFT_MODEL_NAME, NUM_DRAFT_TOKENS, PROMPT_LENGTH


@pytest.mark.parametrize(
    "drafter, hf_model", [(CONFIG_DICT, DRAFT_MODEL_NAME)], indirect=True
)
def test_drafter_one_round(drafter: Drafter, hf_model: Qwen3ForCausalLM) -> None:
    """Verify drafter model outputs match HuggingFace model outputs for one round generation."""
    input_ids = torch.randint(0, hf_model.config.vocab_size, (1, PROMPT_LENGTH))

    drafter._reset()
    draft_outputs, candidate_logits = drafter.draft(input_ids)

    drafter.kv_cache.crop(0)
    hf_outputs = hf_model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=NUM_DRAFT_TOKENS,
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
    hf_logits = torch.stack(hf_outputs.logits, dim=1)
    assert torch.equal(candidate_logits, hf_logits)
