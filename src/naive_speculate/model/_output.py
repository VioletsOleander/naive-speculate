from typing import NamedTuple

import torch


class PrefillOutput(NamedTuple):
    """Prefill output structure.

    Attributes:
        output_ids (torch.Tensor): The updated token IDs after prefill.
        output_logits (torch.Tensor): The logits of newly generated tokens.
    """

    output_ids: torch.Tensor
    output_logits: torch.Tensor


class DecodeOutput(NamedTuple):
    """Decode output structure.

    Attributes:
        output_ids (torch.Tensor): The updated token IDs after decode.
        output_logits (torch.Tensor): The logits of newly generated tokens.
    """

    output_ids: torch.Tensor
    output_logits: torch.Tensor


class OutputCollection:
    """Collection for model outputs during decode and prefill.

    Collects output IDs and optionally collect logits.

    Attributes:
        collect_logits (bool): Whether to collect logits.
        output_ids (torch.Tensor): Collected output IDs.
        output_logits (list[torch.Tensor]): Collected output logits.
    """

    collect_logits: bool
    output_ids: torch.Tensor
    output_logits: list[torch.Tensor]

    def __init__(self, collect_logits: bool) -> None:
        self.collect_logits = collect_logits
        self.output_ids = torch.empty(0)
        self.output_logits = []

    def update(
        self, output_ids: torch.Tensor, output_logits: torch.Tensor | None = None
    ) -> None:
        """Update collected outputs.

        Args:
            output_ids (torch.Tensor): New output IDs to collect.
            output_logits (torch.Tensor | None): New output logits to collect.

        Raises:
            ValueError: If `self.collect_logits` is True but `output_logits` is None.
        """
        self.output_ids = output_ids
        if self.collect_logits:
            if output_logits is None:
                raise ValueError(
                    "output_logits must be provided when collect_logits is True."
                )
            self.output_logits.append(output_logits)

    def trim(self, num_tokens_trim: int) -> None:
        """Trim the last `num_tokens_trim` tokens from collected outputs.

        Args:
            num_tokens_trim (int): Number of tokens to trim

        Raises:
            RuntimeError: If no output IDs collected.
        """
        if self.output_ids.numel() == 0:
            raise RuntimeError("No output IDs collected.")

        self.output_ids = self.output_ids[:, :-num_tokens_trim]

        if self.collect_logits:
            last_collect_logits = self.output_logits[-1]
            self.output_logits[-1] = last_collect_logits[:, :-num_tokens_trim]

    def finalize(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finalize collected outputs and return them.

        If self.collect_logits is False, an empty tensor will be returned as logits.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (output_ids, output_logits)

        Raises:
            RuntimeError: If no output IDs collected.
        """
        if self.output_ids.numel() == 0:
            raise RuntimeError("No output IDs collected.")

        if self.collect_logits:
            return self.output_ids, torch.cat(self.output_logits, dim=1)
        else:
            return self.output_ids, torch.empty(0, device=self.output_ids.device)
