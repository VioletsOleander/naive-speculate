from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch


class SampleStrategy(StrEnum):
    """Sampling strategies for token generation."""

    RANDOM = "random"
    GREEDY = "greedy"


class PrefillOutput(NamedTuple):
    """Prefill output structure.

    Attributes:
        output_ids (torch.Tensor): The newly generated token ids after prefill.
        output_logits (torch.Tensor): The logits of the query tokens (excluding the first ones)
            and the newly generated tokens.
    """

    output_ids: torch.Tensor
    output_logits: torch.Tensor


class DecodeOutput(NamedTuple):
    """Decode output structure.

    Attributes:
        output_ids (torch.Tensor): The newly generated token ids after decode.
        output_logits (torch.Tensor): The logits of newly generated tokens.
    """

    output_ids: torch.Tensor
    output_logits: torch.Tensor
