from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    import torch

    from naive_speculate.infer import KVCache
    from naive_speculate.utils.sample import SampleStrategy


class DraftResult(NamedTuple):
    """Output of `Drafter.draft` method.

    Attributes:
        draft_token_ids: The token ids of the drafted tokens.
            Shape `[batch_size, num_drafted_tokens]`.
        draft_token_logits: The logits corresponding to the drafted tokens.
            Shape `[batch_size, num_drafted_tokens, vocab_size]`.
    """

    draft_token_ids: torch.Tensor
    draft_token_logits: torch.Tensor


class Drafter(Protocol):
    """Drafter is able to generate draft tokens given query tokens and KV cache.

    In the context of speculative decoding, a drafter generates draft tokens
    that are later verified by a more accurate model (the verifier). Typically,
    the drafter is a smaller but faster model than the verifier.
    """

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
        - draft_token_logits: the logits corresponding to the drafted token, of shape
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
        ...
