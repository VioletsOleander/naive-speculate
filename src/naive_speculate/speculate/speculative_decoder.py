from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from .utils import greedy_match, speculative_sample

if TYPE_CHECKING:
    from naive_speculate.draft import Drafter
    from naive_speculate.infer import KVCache
    from naive_speculate.score import Scorer
    from naive_speculate.utils.config import SpeculateConfig
    from naive_speculate.utils.sample import SampleStrategy


class VerifyStrategy(StrEnum):
    """Verification strategies for speculative decoding.

    Attributes:
        GREEDY_MATCH: Verify drafted tokens using greedy matching.
        SPECULATIVE_SAMPLE: Verify drafted tokens using speculative sampling.
    """

    GREEDY_MATCH = "greedy_match"
    SPECULATIVE_SAMPLE = "speculative_sample"


class SpeculativeDecoder:
    """Performs speculative decoding using a drafter and a scorer."""

    _speculate_config: SpeculateConfig
    _drafter: Drafter
    _scorer: Scorer
    _drafter_kvcache: KVCache
    _scorer_kvcache: KVCache

    def __init__(
        self,
        speculate_config: SpeculateConfig,
        drafter: Drafter,
        scorer: Scorer,
        drafter_kvcache: KVCache,
        scorer_kvcache: KVCache,
    ) -> None:
        self._speculate_config = speculate_config
        self._drafter = drafter
        self._scorer = scorer
        self._drafter_kvcache = drafter_kvcache
        self._scorer_kvcache = scorer_kvcache

    @cached_property
    def num_draft_tokens(self) -> int:
        return self._speculate_config.num_draft_tokens

    @cached_property
    def sample_strategy(self) -> SampleStrategy:
        return self._speculate_config.sample_strategy

    def speculative_decode(
        self,
        query_token_ids: torch.Tensor,
        draft_strategy: SampleStrategy,
        verify_strategy: VerifyStrategy,
    ) -> torch.Tensor:
        """Perform speculative decoding.

        Currently supports `batch_size=1` only.

        `draft_strategy` defines how the drafter samples draft tokens.
        `verify_strategy` defines how the drafted tokens are verified.

        The `GREEDY_MATCH` verification strategy is legal to combine with any drafter sampling strategy.

        However, to achieve real speedup, it is suggested to use `SPECULATIVE_SAMPLE` verification
        strategy in favor of `GREEDY_MATCH`, because the latter normally leads to more rejections,
        since it requires exact matches between the drafter's greedy tokens and the target model's greedy tokens.

        Also, greedy decoding normally performs worse than random sampling decoding in terms of generation quality.

        The `SPECULATIVE_SAMPLE` verification strategy is legal to combine with any drafter
        sampling strategy in definition, as long as the sampling strategy defines a valid proposal distribution.

        However, to achieve real speedup, it is suggested to not use `SPECULATIVE_SAMPLE` verification with
        drafter greedy sampling.

        The reason is: If the drafter uses greedy sampling, speculative sampling for verification will
        not be a legal option, because in this case the drafter's distribution is always a delta distribution,
        which makes the acceptance probability ill-defined. In this case, rejection always happens for
        each draft token, and the speculative decoding degenerates to modified auto-regressive decoding,
        with the drafted token being removed from the vocabulary in sampling.

        Args:
            query_token_ids (torch.Tensor): Ids of the query tokens. Shape: `[batch_size, num_query_tokens]`.
            draft_strategy (SampleStrategy): Sampling strategy for drafter.
            verify_strategy (VerifyStrategy): Verification strategy for drafted tokens.

        Returns:
            torch.Tensor: Generated token ids. Shape: `[batch_size, num_generated_tokens]`.
        """
        # 1. Draft
        draft_out = self._drafter.draft(
            query_token_ids=query_token_ids,
            kv_cache=self._drafter_kvcache,
            num_draft_tokens=self.num_draft_tokens,
            sample_strategy=draft_strategy,
        )

        # 2. Score
        context_token_ids = torch.cat([query_token_ids, draft_out.token_ids], dim=1)
        score_out = self._scorer.score(
            query_token_ids=context_token_ids,
            kv_cache=self._scorer_kvcache,
        )

        num_drafted_tokens = draft_out.token_ids.size(1)
        token_scores = score_out.token_logits[:, -(num_drafted_tokens + 1) :, :]

        # 3. Verify (accept and optional resample)
        proposal_dists = torch.softmax(draft_out.token_logits, dim=-1)
        target_dists = torch.softmax(token_scores, dim=-1)

        match verify_strategy:
            case VerifyStrategy.SPECULATIVE_SAMPLE:
                rejected_idx, resampled_token = speculative_sample(
                    target_dists=target_dists.squeeze(0),
                    proposal_dists=proposal_dists.squeeze(0),
                    candidate_tokens=draft_out.token_ids.squeeze(0),
                )
            case VerifyStrategy.GREEDY_MATCH:
                rejected_idx, resampled_token = greedy_match(
                    target_dists=target_dists.squeeze(0),
                    candidate_tokens=draft_out.token_ids.squeeze(0),
                )

        # 4. Crop
        num_tokens_accept = int(rejected_idx.item())
        num_tokens_crop = num_drafted_tokens - num_tokens_accept
        self._drafter_kvcache.crop(num_tokens_crop)
        self._scorer_kvcache.crop(num_tokens_crop)

        # 5. Output
        accepted_tokens = draft_out.token_ids[:, :num_tokens_accept]

        return torch.cat([accepted_tokens, resampled_token.view(1, 1)], dim=1)
