from logging import Logger

import torch
import transformers

from naive_speculate.models import QwenModel
from naive_speculate.utils import SpeculateConfig


class Drafter(QwenModel):
    """Drafter model for generating candidate tokens.

    Attributes:
        draft_tokens_num (int): Number of tokens to draft.
        decode_method (str): Method used for decoding.
        logger (Logger | None): Optional logger for logging information.
    """

    draft_tokens_num: int
    decode_method: str
    logger: Logger | None

    def __init__(self, config: SpeculateConfig, logger: Logger | None = None):
        super().__init__(config.drafter_model_name)
        self.draft_tokens_num = config.draft_tokens_num
        self.decode_method = config.decode_method
        self.logger = logger

    def draft(
        self,
        input_ids: torch.Tensor,
        streamer: transformers.TextStreamer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate candidate tokens given the `input_ids`.

        Returns the generated tokens and their corresponding logits.

        Args:
            input_ids (torch.Tensor): The updated input token IDs.
            streamer (transformers.TextStreamer | None): Optional streamer for real-time output.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the generated token IDs and their logits
        """
        input_ids, candidate_logits = self._prefill(
            input_ids=input_ids,
            decode_method=self.decode_method,
            output_logits=True,
            streamer=streamer,
        )
        new_token_logit = candidate_logits[:, -1:None, :]

        input_ids, candidate_logits = self._decode(
            input_ids=input_ids,
            max_new_tokens=self.draft_tokens_num
            - 1,  # Already generated one token in prefill
            decode_method=self.decode_method,
            output_logits=True,
            streamer=streamer,
        )
        candidate_logits = torch.cat([new_token_logit, candidate_logits], dim=1)

        return input_ids, candidate_logits

    def __str__(self) -> str:
        return f"{super().__str__()}(draft_tokens_num={self.draft_tokens_num})"
