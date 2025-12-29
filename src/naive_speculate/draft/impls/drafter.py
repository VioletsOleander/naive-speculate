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
        - draft_token_logits: the logits corresponding the drafted token, of shape
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
            ValueError: If `self.sample_strategy` is not supported.
        """
        _, num_query_tokens = query_token_ids.size()

        if num_query_tokens <= 0:
            raise ValueError(f"num_query_tokens should be positive, got {num_query_tokens}.")

        if num_query_tokens > 1:
            prefill_out = self.inferencer.prefill(
                query_token_ids=query_token_ids,
                kv_cache=kv_cache,
                sample_strategy=sample_strategy,
            )

            decode_out = self.inferencer.decode(
                query_token_ids=prefill_out.output_ids,
                kv_cache=kv_cache,
                max_new_tokens=num_draft_tokens - 1,
                sample_strategy=sample_strategy,
            )

            draft_token_ids = torch.cat([prefill_out.output_ids, decode_out.output_ids], dim=-1)
            draft_token_logits = torch.cat(
                [prefill_out.output_logits[:, -1:, :], decode_out.output_logits], dim=-2
            )

        else:
            decode_out = self.inferencer.decode(
                query_token_ids=query_token_ids,
                kv_cache=kv_cache,
                max_new_tokens=num_draft_tokens,
                sample_strategy=sample_strategy,
            )

            draft_token_ids = decode_out.output_ids
            draft_token_logits = decode_out.output_logits

        return DraftResult(draft_token_ids, draft_token_logits)
