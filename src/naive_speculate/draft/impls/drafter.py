from typing import TYPE_CHECKING

import torch

from naive_speculate.draft import DraftResult

if TYPE_CHECKING:
    from naive_speculate.infer import Inferencer, KVCache
    from naive_speculate.utils.sample import SampleStrategy


class Drafter:
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

        Return DraftResult, which includes:
        - draft_token_ids: the generated draft token ids, of shape `[batch_size, num_drafted_tokens]`,
            where `num_drafted_tokens <= num_draft_tokens`, because the generation may stop early if
            the end-of-sequence token is generated.
        - draft_token_logits: the logits corresponding to the drafted tokens, of shape
            `[batch_size, num_drafted_tokens, vocab_size]`.

        Args:
            query_token_ids (torch.Tensor): Query tokens of shape `[batch_size, num_query_tokens]`.
            kv_cache (KVCache): Key and value tensors of past tokens.
            num_draft_tokens (int): Limit on the number of tokens to draft.
            sample_strategy (SampleStrategy): The sampling strategy to use during generation.

        Returns:
            DraftResult: A named tuple containing:
                - draft_token_ids (torch.Tensor): The generated draft token ids.
                - draft_token_logits (torch.Tensor): The logits for the drafted tokens.

        Raises:
            ValueError: If `num_query_tokens` is not positive.
            ValueError: If `num_draft_tokens` is not positive.
            ValueError: If `sample_strategy` is not supported.
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
            return DraftResult(
                draft_token_ids=decode_out.token_ids, draft_token_logits=decode_out.token_logits
            )

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
