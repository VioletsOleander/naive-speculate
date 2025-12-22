from logging import Logger

import torch

from naive_speculate.models import QwenModel
from naive_speculate.utils import SpeculateConfig, logger_or_dummy


def greedy_match(
    target_dists: torch.Tensor, candidate_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify candidate tokens against target distributions using greedy matching.

    Args:
        target_dists (torch.Tensor): Target distributions of shape [draft_tokens_num, vocab_size].
        candidate_tokens (torch.Tensor): Candidate sequence of shape [draft_tokens_num].

    Returns:
        rejected_idx (torch.Tensor): Index of the first mismatched token in the sequence. Scalar tensor with empty shape.
          If no rejections happens, equal to `candidate_tokens.shape[0]`, i.e. the number of draft tokens.
        resampled_token (torch.Tensor): Greedy sampled token at the rejected position. Scalar tensor with empty shape.
          If no rejection happens, this will still be a valid tensor but should be ignored.
    """
    assert target_dists.device == candidate_tokens.device
    # target_dist: [draft_tokens_num, vocab_size]
    # candidate_tokens: [draft_tokens_num]
    assert target_dists.shape[0] == candidate_tokens.shape[0]
    draft_tokens_num, vocab_size = target_dists.shape

    # 1. Sample greedy tokens from target distribution
    greedy_tokens = torch.argmax(target_dists, dim=-1)

    # 2. Find the first mismatch position
    matches = greedy_tokens == candidate_tokens
    val, rejected_idx = torch.min(matches, dim=-1)

    # 3. If mismatch happens, get the greedy token at that position
    # No conditional branch here to avoid device synchronization
    resampled_token = greedy_tokens[rejected_idx]

    # if all match, set rejected_idx to draft_tokens_num
    rejected_idx += val * draft_tokens_num
    return rejected_idx, resampled_token


def speculative_sample(
    target_dists: torch.Tensor,
    proposal_dists: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify candidate samples against target distributions using speculative sampling.

    Args:
        target_dists (torch.Tensor): Target distributions of shape [draft_tokens_num, vocab_size].
        proposal_dists (torch.Tensor): Proposal distributions of shape [draft_tokens_num, vocab_size].
        candidate_tokens (torch.Tensor): Candidate tokens of shape [draft_tokens_num].

    Returns:
        rejected_idx (torch.Tensor): Index of the first rejected token. Scalar tensor with empty shape.
          If no rejection happens, equal to `candidate_tokens.shape[0]`, i.e. the number of draft tokens.
        resampled_token (torch.Tensor): Resampled token at the rejected position. Scalar tensor with empty shape.
          If no rejection happens, this will still be a valid tensor but should be ignored.
    """
    assert proposal_dists.device == target_dists.device == candidate_tokens.device
    # proposal_dist, target_dist: [draft_tokens_num, vocab_size]
    # candidate_tokens: [draft_tokens_num]
    assert proposal_dists.shape == target_dists.shape
    assert proposal_dists.shape[0] == candidate_tokens.shape[0]
    draft_tokens_num, vocab_size = proposal_dists.shape

    # 1. Gather probabilities of tokens in candidate_sequences
    proposal_probs = torch.gather(
        proposal_dists, 1, candidate_tokens.unsqueeze(-1)
    ).squeeze(-1)
    target_probs = torch.gather(
        target_dists, 1, candidate_tokens.unsqueeze(-1)
    ).squeeze(-1)

    # 2. Find the first rejection position
    accepted = torch.rand(draft_tokens_num, device=proposal_dists.device) < (
        target_probs / (proposal_probs + 1e-9)
    )
    val, rejected_idx = torch.min(accepted, dim=-1)

    # 3. If rejection happens, resample a token from the residual distribution
    resample_dist = target_dists[rejected_idx] - proposal_dists[rejected_idx]
    resample_dist = (
        torch.clamp_(resample_dist, min=0.0) + 1e-9
    )  # avoid all-zero distribution
    resampled_token = torch.multinomial(resample_dist, num_samples=1).squeeze()

    # if all accepted, set rejected_idx to draft_tokens_num
    rejected_idx += val * draft_tokens_num
    return rejected_idx, resampled_token


class Verifier(QwenModel):
    """Verifier model for speculative decoding.

    Attributes:
        decode_method (str): Decoding method to use during verification.
        first_prefill_done (bool): Flag indicating if the first prefill has been done.
        logger (Logger): Logger for logging information.
    """

    decode_method: str
    first_prefill_done: bool
    logger: Logger

    def __init__(self, config: SpeculateConfig, logger: Logger | None = None) -> None:
        super().__init__(config.verifier_model_name)
        self.decode_method = config.decode_method
        self.first_prefill_done = False
        self.logger = logger_or_dummy(logger)

    @torch.no_grad()
    def verify(
        self, input_ids: torch.Tensor, proposal_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Verify the candidate samples generated by the draft model.

        Args:
            input_ids (torch.Tensor): The input token IDs for the verifier model. Shape: [batch_size, seq_len].
              Normally, just pass the output_ids generated by the drafter model's `drafter` method to it.
            proposal_logits (torch.Tensor): Logits of the candidate sequences generated by the draft model. Shape: [batch_size, draft_tokens_num, vocab_size].
              Normally, just pass the output_logits generated by the drafter model's `drafter` method to it.

        Returns:
            rejected_idx (torch.Tensor): Index of the first rejected token in the candidate sequence. Shape: Scalar tensor with empty shape.
              If no rejection happens, equal to `draft_tokens_num`, i.e. the number of draft tokens.
            resampled_token (torch.Tensor): Resampled token at the rejected position. Shape: Scalar tensor with empty shape.
              If no rejection happens, this will an additional new token.

        Raises:
            ValueError: If `self.decode_method` is not supported.
        """
        draft_len = proposal_logits.shape[1]

        # 1. Prefill
        if not self.first_prefill_done:
            verify_input_ids = input_ids
            self.first_prefill_done = True
        else:
            verify_input_ids = input_ids[:, -draft_len:]

        output_ids, output_logits = self._prefill(
            input_ids=verify_input_ids,
            decode_method=self.decode_method,
            output_logits=True,
        )

        # 2. Verify
        target_dists = output_logits[0, -draft_len - 1 : -1]
        proposal_dists = proposal_logits[0]
        candidate_tokens = verify_input_ids[0, -draft_len:]

        match self.decode_method:
            case "greedy":
                rejected_idx, resampled_token = greedy_match(
                    target_dists=target_dists,
                    candidate_tokens=candidate_tokens,
                )
            case "random":
                rejected_idx, resampled_token = speculative_sample(
                    target_dists=target_dists,
                    proposal_dists=proposal_dists,
                    candidate_tokens=candidate_tokens,
                )
            case _:
                raise ValueError(f"Unsupported decode method: {self.decode_method}")

        # if no rejection, use the last token from output_ids
        no_rejection = rejected_idx == draft_len
        resampled_token = (
            resampled_token * ~no_rejection + output_ids[0, -1] * no_rejection
        )

        return rejected_idx, resampled_token

    def _reset(self) -> None:
        """Reset the verifier state for new generation session."""
        self.first_prefill_done = False
