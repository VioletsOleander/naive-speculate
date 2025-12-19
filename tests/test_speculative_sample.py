import pytest
import torch

from naive_speculate.verify import speculative_sample

NUM_SAMPLES = 8000
CHECK_INTERVAL = 500


@pytest.fixture
def target_and_proposal_dists(
    request: pytest.FixtureRequest,
) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_size = request.param[0]
    draft_length = request.param[1]
    assert isinstance(draft_length, int)
    assert isinstance(vocab_size, int)

    target_dists = torch.softmax(torch.randn(draft_length, vocab_size), dim=-1)
    proposal_dists = torch.softmax(torch.randn(draft_length, vocab_size), dim=-1)
    return target_dists, proposal_dists


def compute_kl_divergences(
    target_dists: torch.Tensor, empirical_dists: torch.Tensor
) -> torch.Tensor:
    assert target_dists.shape == empirical_dists.shape
    assert target_dists.dim() == 2  # [draft_length, vocab_size]

    kl_divergences = torch.sum(
        target_dists * (torch.log(target_dists) - torch.log(empirical_dists + 1e-10)),
        dim=-1,
    )

    return kl_divergences  # [draft_length]


# TODO: Marginal case is ok now, but joint case needs more investigation,
# maybe a different formulation is needed
@pytest.mark.parametrize(
    "target_and_proposal_dists",
    [
        (50, 1),
        pytest.param(
            (10, 5),
            marks=pytest.mark.xfail(
                reason="Joint case needs more investigation",  # now normally this will pass
            ),
        ),
    ],  # (vocab_size, draft_length)
    ids=["marginal", "joint"],
    indirect=True,
)
def test_speculative_sample(
    target_and_proposal_dists: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify that speculative sampling approximates the target distribution over time."""
    target_dists, proposal_dists = target_and_proposal_dists
    assert target_dists.shape == proposal_dists.shape
    draft_length, vocab_size = proposal_dists.shape

    kl_divergences_list = []
    # Notice that original kl divergences sometimes much lower than empirical ones
    # further exploration needed
    original_kl_divergences = compute_kl_divergences(target_dists, proposal_dists)

    # [draft_length, NUM_SAMPLES] -> [NUM_SAMPLES, draft_length]
    candidate_sequences = torch.multinomial(
        proposal_dists, num_samples=NUM_SAMPLES, replacement=True
    ).mT
    token_frequencies = torch.zeros(draft_length, vocab_size)

    for sample_idx in range(NUM_SAMPLES):
        candidate_tokens = candidate_sequences[sample_idx]  # [draft_length]
        # 1. Speculative sampling
        rejected_idx, resampled_token = speculative_sample(
            proposal_dists=proposal_dists,
            target_dists=target_dists,
            candidate_tokens=candidate_tokens,
        )

        # 2. Collect accepted tokens, update frequencies
        accepted_tokens = candidate_tokens[:rejected_idx]
        if rejected_idx < draft_length:
            accepted_tokens = torch.cat([accepted_tokens, resampled_token], dim=0)
        accepted_tokens = accepted_tokens.tolist()
        for pos, token_idx in enumerate(accepted_tokens):
            token_frequencies[pos, token_idx] += 1

        # 3. Compute empirical distribution and compare to target
        if (sample_idx + 1) % CHECK_INTERVAL == 0:
            # normalize by the actual number of visits to each position
            position_visits = token_frequencies.sum(
                dim=-1, keepdim=True
            )  # [draft_length, 1]
            empirical_dists = token_frequencies / position_visits

            kl_divergences = compute_kl_divergences(target_dists, empirical_dists)
            kl_divergences_list.append(kl_divergences)

    # Check that KL divergences decrease over time
    # [NUM_SAMPLES/CHECK_INTERVAL, draft_length] -> [draft_length, NUM_SAMPLES/CHECK_INTERVAL]
    kl_divergences_list_tensor = torch.stack(kl_divergences_list).mT
    diffs = kl_divergences_list_tensor[:, 1:] - kl_divergences_list_tensor[:, :-1]
    assert torch.all(diffs <= 0.1)  # allow small numerical fluctuations
    assert torch.all(diffs.sum(dim=-1) < 0.0)  # overall must decrease
    assert torch.all(
        kl_divergences_list_tensor[:, -1] < original_kl_divergences
    )  # final kl divergence must be lower than original
