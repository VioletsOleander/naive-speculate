"""Define `Drafter` class, implementing token drafting functionality."""

from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from naive_speculate.config.strategy import SampleStrategy
    from naive_speculate.infer import Inferencer, KVCache

__all__ = ["DraftResult", "Drafter"]


class DraftResult(NamedTuple):
    """Output of `Drafter.draft` method.

    Attributes:
        token_ids: The token ids of the drafted tokens.
            Shape `[batch_size, num_drafted_tokens]`.
        token_logits: The logits used to sample the drafted tokens.
            Shape `[batch_size, num_drafted_tokens, vocab_size]`.
    """

    token_ids: torch.Tensor
    token_logits: torch.Tensor


class Drafter:
    """Drafter is able to generate draft tokens given query tokens and KV cache.

    Drafter delegates token drafting to an `Inferencer` instance.

    In the context of speculative decoding, a drafter generates draft tokens
    that are later verified by a more accurate model (the verifier). Typically,
    the drafter is a smaller but faster model than the verifier.

    Attributes:
        inferencer (Inferencer): The inferencer used for drafting tokens.
    """

    inferencer: Inferencer

    def __init__(self, inferencer: Inferencer) -> None:
        self.inferencer = inferencer

    @torch.no_grad()
    def draft(
        self,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
        num_draft_tokens: int,
        sample_strategy: SampleStrategy,
    ) -> DraftResult:
        """Generate candidate tokens given query tokens and KV cache.

        `kv_cache` will be updated internally as a side effect of this method.

        Return DraftResult, which includes:
        - token_ids: the generated draft token ids, of shape `[batch_size, num_drafted_tokens]`,
            where `num_drafted_tokens <= num_draft_tokens`, because the generation may stop early if
            the end-of-sequence token is generated.
        - token_logits: the logits used to sample the drafted token, of shape
            `[batch_size, num_drafted_tokens, vocab_size]`.

        `num_query_tokens := query_token_ids.shape[1]` is expected to be positive.
        `num_draft_tokens` is expected to be positive.

        Args:
            query_token_ids (torch.Tensor): Query tokens of shape `[batch_size, num_query_tokens]`.
            kv_cache (KVCache): Key and value tensors of past tokens.
            num_draft_tokens (int): Limit on the number of tokens to draft, should be positive.
            sample_strategy (SampleStrategy): The sampling strategy to use during generation.

        Returns:
            DraftResult: A named tuple containing:
                - token_ids (torch.Tensor): The generated draft token ids.
                - token_logits (torch.Tensor): The logits for the drafted tokens.
        """
        if num_draft_tokens <= 0:
            raise ValueError(f"num_draft_tokens should be positive, got {num_draft_tokens}.")

        if (num_query_tokens := query_token_ids.size(1)) <= 0:
            raise ValueError(f"num_query_tokens should be positive, got {num_query_tokens}.")

        # Decode only path
        if num_query_tokens == 1:
            decode_out = self.inferencer.decode(
                query_token_ids=query_token_ids,
                kv_cache=kv_cache,
                max_new_tokens=num_draft_tokens,
                sample_strategy=sample_strategy,
            )
            return DraftResult(token_ids=decode_out.token_ids, token_logits=decode_out.token_logits)

        # Prefill + optional decode path
        # 1. Prefill
        prefill_out = self.inferencer.prefill(
            query_token_ids=query_token_ids,
            kv_cache=kv_cache,
            sample_strategy=sample_strategy,
        )
        draft_token_ids = prefill_out.token_ids[:, -1:]
        draft_token_logits = prefill_out.token_logits[:, -1:, :]

        # 2. Optional decode
        if num_draft_tokens > 1:
            decode_out = self.inferencer.decode(
                query_token_ids=draft_token_ids,
                kv_cache=kv_cache,
                max_new_tokens=num_draft_tokens - 1,
                sample_strategy=sample_strategy,
            )
            draft_token_ids = torch.cat([draft_token_ids, decode_out.token_ids], dim=1)
            draft_token_logits = torch.cat([draft_token_logits, decode_out.token_logits], dim=1)

        return DraftResult(draft_token_ids, draft_token_logits)
