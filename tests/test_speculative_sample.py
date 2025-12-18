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

    target_dists = torch.softmax(torch.randn(1, draft_length, vocab_size), dim=-1)
    proposal_dists = torch.softmax(torch.randn(1, draft_length, vocab_size), dim=-1)
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
                reason="Joint case needs more investigation",
            ),
        ),
    ],  # (vocab_size, draft_length)
    ids=["marginal", "joint"],
    indirect=True,
)
def test_speculative_sample(
    target_and_proposal_dists: tuple[torch.Tensor, torch.Tensor],
) -> None:
    target_dists, proposal_dists = target_and_proposal_dists
    assert target_dists.shape == proposal_dists.shape
    batch_size, draft_length, vocab_size = proposal_dists.shape
    assert batch_size == 1

    kl_divergences_list = []
    # Notice that original kl divergences sometimes much lower than empirical ones
    # further exploration needed
    original_kl_divergences = compute_kl_divergences(
        target_dists.squeeze(0), proposal_dists.squeeze(0)
    )
    # kl_divergences_list.append(original_kl_divergences)

    # [draft_length, NUM_SAMPLES] -> [1, NUM_SAMPLES, draft_length]
    candidate_samples = torch.multinomial(
        proposal_dists.squeeze(0), num_samples=NUM_SAMPLES, replacement=True
    ).mT.unsqueeze(0)
    token_frequencies = torch.zeros(draft_length, vocab_size)

    for sample_idx in range(NUM_SAMPLES):
        # 1. Speculative sampling
        candidate_sequences = candidate_samples[:, sample_idx]  # [1, draft_length]
        rejected_idx, resampled_tokens = speculative_sample(
            proposal_dists=proposal_dists,
            target_dists=target_dists,
            candidate_sequences=candidate_sequences,
        )

        # 2. Collect accepted tokens, update frequencies
        accepted_tokens = candidate_sequences[:, :rejected_idx].squeeze(0)
        if resampled_tokens is not None:
            resampled_tokens.squeeze_(0)
            accepted_tokens = torch.cat([accepted_tokens, resampled_tokens], dim=0)
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

            kl_divergences = compute_kl_divergences(
                target_dists.squeeze(0), empirical_dists
            )
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
