import pytest
import torch

from naive_speculate.models import QwenModel
from tests.models.constants import MAX_NEW_TOKENS, MODEL_NAME, PROMPT_LENGTH


@pytest.mark.parametrize("custom_model", [MODEL_NAME], indirect=True)
def test_inference_consistency(custom_model: QwenModel) -> None:
    """Verify that the model produces consistent outputs across multiple runs with the same input."""
    input_ids = torch.randint(
        0, custom_model.model.config.vocab_size, (1, PROMPT_LENGTH)
    )

    custom_model._reset()
    output1 = custom_model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        decode_method="greedy",
    )

    custom_model._reset()
    output2 = custom_model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        decode_method="greedy",
    )

    assert torch.equal(output1, output2)
