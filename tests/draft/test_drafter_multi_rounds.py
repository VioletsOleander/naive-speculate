import pytest
import torch
from transformers import Qwen3ForCausalLM
from transformers.generation.utils import GenerateDecoderOnlyOutput

from naive_speculate.draft import Drafter

from .constants import (
    CONFIG_DICT,
    DRAFT_MODEL_NAME,
    NUM_DRAFT_TOKENS,
    MAX_NEW_TOKENS,
    PROMPT_LENGTH,
)


class DrafterOutputs:
    ids: torch.Tensor
    logits: torch.Tensor

    def __init__(self) -> None:
        self.ids = torch.tensor([])
        self.logits = torch.tensor([])


@pytest.mark.parametrize(
    "drafter, hf_model", [(CONFIG_DICT, DRAFT_MODEL_NAME)], indirect=True
)
def test_drafter_multi_rounds(drafter: Drafter, hf_model: Qwen3ForCausalLM) -> None:
    """Verify drafter model outputs match HuggingFace model outputs for multiple rounds generation."""
    input_ids = torch.randint(0, hf_model.config.vocab_size, (1, PROMPT_LENGTH))

    drafter._reset()

    drafter_outputs = DrafterOutputs()
    drafter_inputs = input_ids
    num_total_tokens = input_ids.shape[1]

    while True:
        if num_total_tokens >= PROMPT_LENGTH + MAX_NEW_TOKENS:
            break

        draft_ids, draft_logits = drafter.draft(drafter_inputs)

        # simulate rejection
        rejected_idx = torch.randint(0, NUM_DRAFT_TOKENS + 1, (1,)).item()
        assert isinstance(rejected_idx, int)
        assert 0 <= rejected_idx <= NUM_DRAFT_TOKENS

        # remove the rejected tokens
        num_total_tokens = draft_ids.shape[1]
        if rejected_idx == NUM_DRAFT_TOKENS:
            num_tokens_remove = 0
        else:
            num_tokens_remove = NUM_DRAFT_TOKENS - rejected_idx - 1
        num_total_tokens -= num_tokens_remove

        # note that kv cache does not keep for the last token,
        # since we treat it as the newly generated one
        drafter.kv_cache.crop(num_total_tokens - 1)
        if num_tokens_remove > 0:
            draft_ids = draft_ids[:, :-num_tokens_remove]
            draft_logits = draft_logits[:, :-num_tokens_remove, :]

        drafter_inputs = draft_ids

        drafter_outputs.ids = draft_ids
        drafter_outputs.logits = torch.cat(
            (drafter_outputs.logits, draft_logits), dim=1
        )

    drafter.kv_cache.crop(0)
    hf_outputs = hf_model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        past_key_values=drafter.kv_cache,
        output_logits=True,
        logits_to_keep=0,  # keep all logits
        return_dict_in_generate=True,
    )
    assert isinstance(hf_outputs, GenerateDecoderOnlyOutput)
    hf_output_len = hf_outputs.sequences.shape[1]
    assert torch.equal(hf_outputs.sequences, drafter_outputs.ids[:, :hf_output_len])

    assert hf_outputs.logits is not None
    hf_logits = torch.stack(hf_outputs.logits, dim=1)
    hf_output_len = hf_logits.shape[1]
    assert torch.equal(hf_logits, drafter_outputs.logits[:, :hf_output_len, :])
