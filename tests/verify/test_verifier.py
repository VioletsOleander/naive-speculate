import pytest
import torch

from naive_speculate.draft import Drafter
from naive_speculate.verify import Verifier

DRAFT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DRAFT_TOKENS_NUM = 20
CONFIG_DICT = {
    "max_new_tokens": 32768,
    "decode_method": "greedy",
    "drafter_model_name": DRAFT_MODEL_NAME,
    "draft_tokens_num": DRAFT_TOKENS_NUM + 1,
    "verifier_model_name": DRAFT_MODEL_NAME,
}
PROMPT_LENGTH = 16
VOCAB_SIZE = 64


@pytest.mark.parametrize(
    "verifier, drafter",
    [(CONFIG_DICT, CONFIG_DICT)],
    indirect=True,
)
def test_verifier(verifier: Verifier, drafter: Drafter):
    """Verify that the verifier works as expected when drafter model is the same as verify model."""
    input_ids = torch.randint(low=0, high=VOCAB_SIZE, size=(1, PROMPT_LENGTH))

    drafter._reset()
    drafter.kv_cache.crop(0)
    verifier._reset()
    verifier.kv_cache.crop(0)
    drafter_output_ids, drafter_output_logits = drafter.draft(input_ids)
    rejected_idx, resampled_token = verifier.verify(
        drafter_output_ids[:, :-1],
        drafter_output_logits[:, :-1],
    )

    assert rejected_idx.shape == torch.Size()
    assert resampled_token.shape == torch.Size()
    assert rejected_idx.item() == DRAFT_TOKENS_NUM
    assert resampled_token.item() == drafter_output_ids[0, -1].item()
