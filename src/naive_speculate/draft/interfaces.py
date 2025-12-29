from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    import torch

    from naive_speculate.utils.sample import SampleStrategy


class DraftResult(NamedTuple):
    """The result of a drafting operation.

    Attributes:
        draft_token_ids: The token IDs of the drafted tokens. Shape `[batch_size, num_draft_tokens]`.
        draft_token_logits: The logits corresponding to the drafted tokens. Shape `[batch_size, num_draft_tokens, vocab_size]`.
    """

    draft_token_ids: torch.Tensor
    draft_token_logits: torch.Tensor


class Drafter(Protocol):
    """Drafter is able to generate draft tokens based on a given context.

    In the context of speculative decoding, a drafter generates draft tokens
    that are later verified by a more accurate model (the verifier). Typically,
    the drafter is a smaller but faster model than the verifier.
    """

    def draft(
        self,
        context_token_ids: torch.Tensor,
        num_draft_tokens: int,
        sample_strategy: SampleStrategy,
    ) -> DraftResult:
        """Generate draft tokens based on the given context.

        Args:
            context_token_ids: A tensor of shape `[batch_size, context_length]`
                containing the token IDs representing the context.
            num_draft_tokens: The number of draft tokens to generate.
            sample_strategy: The strategy to use for sampling draft tokens.

        Returns:
            DraftResult: The result containing the drafted token IDs and their logits.
        """
        ...
