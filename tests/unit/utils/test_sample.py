import pytest
import torch

from naive_speculate.config.strategy import SampleStrategy
from naive_speculate.utils.sample import sample_tokens


@pytest.mark.parametrize(
    ("token_logits", "expected"),
    [
        (torch.tensor([[0.1, 0.9], [0.8, 0.2]]), torch.tensor([[1], [0]])),
        (torch.tensor([[0.6, 0.4], [0.3, 0.7]]), torch.tensor([[0], [1]])),
    ],
)
def test_sample_tokens_greedy(token_logits: torch.Tensor, expected: torch.Tensor) -> None:
    sampled = sample_tokens(token_logits, SampleStrategy.GREEDY)
    assert torch.equal(sampled, expected), f"Expected {expected}, but got {sampled}"


@pytest.mark.parametrize(
    ("token_logits", "expected_shape"),
    [
        (torch.tensor([[0.0, 1.0], [1.0, 0.0]]), torch.Size([2, 1])),
        (torch.tensor([[0.5, 0.0], [0.0, 0.5]]), torch.Size([2, 1])),
    ],
)
def test_sample_tokens_random(token_logits: torch.Tensor, expected_shape: torch.Size) -> None:
    sampled = sample_tokens(token_logits, SampleStrategy.RANDOM)
    assert sampled.shape == expected_shape, (
        f"Expected shape {expected_shape}, but got {sampled.shape}"
    )
