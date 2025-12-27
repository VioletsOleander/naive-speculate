from enum import StrEnum
from typing import NamedTuple

import torch


class SampleStrategy(StrEnum):
    """Sampling strategies for token generation."""

    RANDOM = "random"
    GREEDY = "greedy"


class PrefillOutput(NamedTuple):
    """Prefill output structure.

    Attributes:
        output_ids (torch.Tensor): The updated token ids after prefill.
        output_logits (torch.Tensor): The logits of newly generated tokens.
    """

    output_ids: torch.Tensor
    output_logits: torch.Tensor


class DecodeOutput(NamedTuple):
    """Decode output structure.

    Attributes:
        output_ids (torch.Tensor): The updated token ids after decode.
        output_logits (torch.Tensor): The logits of newly generated tokens.
    """

    output_ids: torch.Tensor
    output_logits: torch.Tensor
