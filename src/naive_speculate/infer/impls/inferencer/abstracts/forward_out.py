from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class ForwardOutput(NamedTuple):
    """Output of `BaseInferencer._forward` method.

    Attributes:
        logits (torch.Tensor): Computed logits of shape `[batch_size, num_query_tokens, vocab_size]`.
        keys (Sequence[torch.Tensor]): Key tensors corresponding to the query tokens.
        values (Sequence[torch.Tensor]): Value tensors corresponding to the query tokens.
    """

    logits: torch.Tensor
    keys: Sequence[torch.Tensor]
    values: Sequence[torch.Tensor]
