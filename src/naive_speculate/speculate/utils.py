import torch


def _indexing_or_mask(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index `tensor` at positions `idx`, return zero if `idx` is out of bounds."""
    clampped_idx = torch.clamp(idx, max=tensor.size(0))
    indexed_tensor = tensor[clampped_idx]

    return indexed_tensor * (idx < tensor.size(0))


def speculative_sample(
    target_dists: torch.Tensor,
    proposal_dists: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify candidate samples against target distributions using speculative sampling.

    Expect `target_dists` and `proposal_dists` to be proper probability distributions,
    instead of raw logits.

    `target_dists` should have shape `[num_draft_tokens + 1, vocab_size]`, where the
    extra distribution at the end is for sampling new possible tokens if rejection happens.

    If rejection happens at position `i`, the token for position `i` will be resampled from
    the residual distribution.

    Args:
        target_dists (torch.Tensor): Target distributions of shape [num_draft_tokens + 1, vocab_size].
        proposal_dists (torch.Tensor): Proposal distributions of shape [num_draft_tokens, vocab_size].
        candidate_tokens (torch.Tensor): Candidate tokens of shape [num_draft_tokens].

    Returns:
        rejected_idx (torch.Tensor): Index of the first rejected token. Scalar tensor with empty shape.
            Range: `[0, num_draft_tokens]`.  If no rejection happens,
            equal to `num_draft_tokens`.
        resampled_token (torch.Tensor): Resampled token at the rejected position. Scalar tensor with empty shape.
            If no rejection happens, this will be the token sampled from the extra distribution
            at the end of `target_dists`.
    """
    num_draft_tokens = candidate_tokens.size(0)

    # 1. Gather probabilities of tokens in candidate_sequences
    proposal_probs = torch.gather(proposal_dists, 1, candidate_tokens.unsqueeze(-1)).squeeze(-1)
    target_probs = torch.gather(target_dists[:-1], 1, candidate_tokens.unsqueeze(-1)).squeeze(-1)

    # 2. Find the first rejection position
    accepted = torch.rand(num_draft_tokens, device=proposal_dists.device) < (
        target_probs / (proposal_probs + 1e-9)
    )
    val, rejected_idx = torch.min(accepted, dim=-1)
    rejected_idx += val * num_draft_tokens  # if all accepted, rejected_idx == draft_tokens_num

    # 3. Resample/Sample a token from the residual/extra distribution
    resample_dist = target_dists[rejected_idx] - _indexing_or_mask(proposal_dists, rejected_idx)
    resample_dist = torch.clamp_(resample_dist, min=0.0) + 1e-9  # avoid all-zero distribution
    resampled_token = torch.multinomial(resample_dist, num_samples=1).squeeze()

    return rejected_idx, resampled_token


def greedy_match(
    target_dists: torch.Tensor, candidate_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify candidate tokens against target distributions using greedy matching.

    Expect `target_dists` to be proper probability distributions, instead of raw logits.

    `target_dists` should have shape `[num_draft_tokens + 1, vocab_size]`, where the
    extra distribution at the end is for sampling new possible tokens if rejection happens.

    If rejection happens at position `i`, the token for position `i` will be resampled from
    the target distribution.

    Args:
        target_dists (torch.Tensor): Target distributions of shape [num_draft_tokens + 1, vocab_size].
        candidate_tokens (torch.Tensor): Candidate sequence of shape [num_draft_tokens].

    Returns:
         rejected_idx (torch.Tensor): Index of the first rejected token. Scalar tensor with empty shape.
             Range: `[0, num_draft_tokens]`.  If no rejection happens,
             equal to `num_draft_tokens`.
         resampled_token (torch.Tensor): Resampled token at the rejected position. Scalar tensor with empty shape.
             If no rejection happens, this will be the token sampled from the extra distribution
             at the end of `target_dists`.
    """
    num_draft_tokens = candidate_tokens.size(0)

    # 1. Sample greedy tokens from target distribution
    greedy_tokens = torch.argmax(target_dists, dim=-1)

    # 2. Find the first mismatch position
    matches = greedy_tokens == candidate_tokens
    val, rejected_idx = torch.min(matches, dim=-1)
    rejected_idx += val * num_draft_tokens  # if all accepted, rejected_idx == draft_tokens_num

    # 3. Get the greedy token at the rejected position or the extra token at the end
    resampled_token = greedy_tokens[rejected_idx]

    return rejected_idx, resampled_token
