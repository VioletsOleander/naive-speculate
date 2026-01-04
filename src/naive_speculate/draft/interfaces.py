from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    import torch

    from naive_speculate.infer import KVCache
    from naive_speculate.utils.config import SampleStrategy


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


class Drafter(Protocol):
    """Drafter is able to generate draft tokens given query tokens and KV cache.

    Drafter can either be itself an auto-regressive language model, or delegate
    the generation to an underlying language model.

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
        ...
