from logging import Logger

import torch

from naive_speculate.models import QwenModel
from naive_speculate.utils import SpeculateConfig


def greedy_match(
    target_dists: torch.Tensor, candidate_sequences: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Verify candidate samples against target distributions using greedy matching.

    Args:
        target_dists (torch.Tensor): Target distributions of shape [batch_size, draft_tokens_num, vocab_size].
        candidate_sequences (torch.Tensor): Candidate sequences of shape [batch_size, draft_tokens_num].

    Returns:
        rejected_idx (torch.Tensor): Indices of the first mismatched token for each batch. Shape: [batch_size].
        equivalent to `candidate_sequences.shape[-1]`, i.e. the number of draft tokens, if no mismatch happens.
        resampled_tokens (torch.Tensor | None): Greedy sampled tokens at the rejected positions. Shape: [batch_size].
        equivalent to None if no mismatch happens.
    """
    assert target_dists.device == candidate_sequences.device
    # target_dist: [batch_size, draft_tokens_num, vocab_size]
    # candidate_sample: [batch_size, draft_tokens_num]
    assert target_dists.shape[:-1] == candidate_sequences.shape
    batch_size, draft_tokens_num, vocab_size = target_dists.shape

    # 1. Sample greedy tokens from target distribution
    greedy_tokens = torch.argmax(target_dists, dim=-1)

    # 2. Find the first mismatch position
    matches = greedy_tokens == candidate_sequences
    val, rejected_idx = torch.min(matches, dim=-1)
    # if all match, set rejected_idx to draft_tokens_num
    rejected_idx += val * draft_tokens_num
    rejected_idx.squeeze_()

    # 3. If mismatch happens, get the greedy token at that position
    resampled_tokens = None
    if rejected_idx != draft_tokens_num:
        resampled_tokens = greedy_tokens[:, rejected_idx]

    return rejected_idx, resampled_tokens


def speculative_sample(
    proposal_dists: torch.Tensor,
    target_dists: torch.Tensor,
    candidate_sequences: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Verify candidate samples against target distributions using speculative sampling.

    Args:
        proposal_dists (torch.Tensor): Proposal distributions of shape [batch_size, draft_tokens_num, vocab_size].
        target_dists (torch.Tensor): Target distributions of shape [batch_size, draft_tokens_num, vocab_size].
        candidate_sequences (torch.Tensor): Candidate sequences of shape [batch_size, draft_tokens_num].

    Returns:
        rejected_idx (torch.Tensor): Indices of the first rejected token for each batch. Shape: [batch_size].
        equivalent to `candidate_sequences.shape[-1]`, i.e. the number of draft tokens, if no rejection happens.
        resampled_tokens (torch.Tensor | None): Resampled tokens at the rejected positions. Shape: [batch_size].
        equivalent to None if no rejection happens.
    """
    assert proposal_dists.device == target_dists.device == candidate_sequences.device
    # proposal_dist, target_dist: [batch_size, draft_tokens_num, vocab_size]
    # candidate_sample: [batch_size, draft_tokens_num]
    assert proposal_dists.shape == target_dists.shape
    assert proposal_dists.shape[:-1] == candidate_sequences.shape
    batch_size, draft_tokens_num, vocab_size = proposal_dists.shape

    # 1. Gather probabilities of tokens in candidate_sequences
    proposal_probs = torch.gather(
        proposal_dists, 2, candidate_sequences.unsqueeze(-1)
    ).squeeze(-1)
    target_probs = torch.gather(
        target_dists, 2, candidate_sequences.unsqueeze(-1)
    ).squeeze(-1)

    # 2. Find the first rejection position
    accepted = torch.rand(
        batch_size, draft_tokens_num, device=proposal_dists.device
    ) < (target_probs / proposal_probs)
    val, rejected_idx = torch.min(accepted, dim=-1)
    # if all accepted, set rejected_idx to draft_tokens_num
    rejected_idx += val * draft_tokens_num
    rejected_idx.squeeze_()

    resampled_tokens = None
    # 3. If rejection happens, resample a token from the residual distribution
    if rejected_idx != draft_tokens_num:
        resample_dist = target_dists[:, rejected_idx] - proposal_dists[:, rejected_idx]
        resample_dist = torch.clamp_(resample_dist, min=0.0)
        resampled_tokens = torch.multinomial(resample_dist, num_samples=1)

    return rejected_idx, resampled_tokens


class Verifier(QwenModel):
    logger: Logger | None
    decode_method: str

    def __init__(self, config: SpeculateConfig, logger: Logger | None = None) -> None:
        super().__init__(config.verifier_model_name)
        self.decode_method = config.decode_method
        self.logger = logger

    @torch.no_grad()
    def verify(self, draft_ids: torch.Tensor, draft_logits: torch.Tensor):
        output_ids, output_logits = self._prefill(
            input_ids=draft_ids, decode_method=self.decode_method, output_logits=True
        )
        print("shape of draft_ids:", draft_ids.shape)
        print("shape of output_ids:", output_ids.shape)
        print("shape of output_logits:", output_logits.shape)
        print("shape of draft_logits:", draft_logits.shape)
