import pytest
import torch

from naive_speculate.draft import Drafter
from naive_speculate.verify import Verifier

DRAFT_MODEL_NAME = "Qwen/Qwen3-0.6B"
CONFIG_DICT = {
    "max_new_tokens": 32768,
    "decode_method": "greedy",
    "drafter_model_name": DRAFT_MODEL_NAME,
    "draft_tokens_num": 20,
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
    """Verify that the verifier works as expected whtn drafter model is the same as verify model."""
    input_ids = torch.randint(low=0, high=VOCAB_SIZE, size=(1, PROMPT_LENGTH))

    drafter.kv_cache.crop(0)
    verifier.kv_cache.crop(0)
    drafter_output_ids, drafter_output_logits = drafter.draft(input_ids)
    verifier_output_ids = verifier.verify(drafter_output_ids, drafter_output_logits)

    batch_size, seq_len = drafter_output_ids.shape
    assert verifier_output_ids.shape == (batch_size, seq_len + 1)
    assert torch.all(verifier_output_ids[:, :-1] == drafter_output_ids)
