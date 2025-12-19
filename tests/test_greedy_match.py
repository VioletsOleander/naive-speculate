import pytest
import torch

from naive_speculate.verify import greedy_match

VOCAB_SIZE = 10
DRAFT_LENGTH = 5


@pytest.fixture
def target_dists() -> torch.Tensor:
    return torch.softmax(torch.randn(DRAFT_LENGTH, VOCAB_SIZE), dim=-1)


def test_greedy_match(target_dists: torch.Tensor) -> None:
    target_preds = torch.argmax(target_dists, dim=-1)

    # Test cases where tokens match up to a certain index
    for reject_idx in range(DRAFT_LENGTH):
        candidate_preds = torch.ones_like(target_preds) * VOCAB_SIZE
        candidate_preds[0:reject_idx] = target_preds[0:reject_idx]

        rejected_idx, resampled_token = greedy_match(target_dists, candidate_preds)
        assert rejected_idx == reject_idx
        assert torch.equal(resampled_token, target_preds[reject_idx])

    # Test case where all tokens match
    candidate_preds = target_preds.clone()
    rejected_idx, resampled_token = greedy_match(
        target_dists=target_dists, candidate_tokens=candidate_preds
    )
    assert rejected_idx == DRAFT_LENGTH
