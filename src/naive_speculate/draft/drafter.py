from logging import Logger

import torch

from naive_speculate.models import QwenModel
from naive_speculate.utils import SpeculateConfig, logger_or_dummy


class Drafter(QwenModel):
    """Drafter model for generating candidate tokens.

    Attributes:
        draft_tokens_num (int): Number of tokens to draft in each round.
        decode_method (str): Should be 'greedy' or 'random'.
        prefill_done (bool): If prefill has been done.
        logger (Logger): Used for logging.
    """

    draft_tokens_num: int
    decode_method: str
    prefill_done: bool
    logger: Logger

    def __init__(self, config: SpeculateConfig, logger: Logger | None = None):
        super().__init__(config.drafter_model_name)
        self.draft_tokens_num = config.draft_tokens_num
        self.decode_method = config.decode_method
        self.prefill_done = False
        self.logger = logger_or_dummy(logger)

    @torch.no_grad()
    def draft(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate candidate tokens given the context.

        On first call, performs prefill. On subsequent calls, continues generation
        from previous state.
        Returns the updated tokens IDs and logits of newly generated tokens.

        Args:
            input_ids (torch.Tensor): The context's token IDs. Shape: `[batch_size, seq_len]`

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The updated token IDs and logits of the generated draft tokens.
            Shapes:
              - updated_input_ids: `[batch_size, seq_len + draft_tokens_num]`
              - candidate_logits: `[batch_size, draft_tokens_num, vocab_size]`

        Raises:
            ValueError: If `self.decode_method` is not supported.
        """
        if not self.prefill_done:
            input_ids, candidate_logits = self._prefill(
                input_ids=input_ids,
                decode_method=self.decode_method,
                output_logits=True,
            )
            new_token_logit = candidate_logits[:, -1:, :]

        # bug here, requires fix
        input_ids, candidate_logits = self._decode(
            input_ids=input_ids,
            max_new_tokens=self.draft_tokens_num
            - 1,  # Already generated one token in prefill
            decode_method=self.decode_method,
            output_logits=True,
        )

        if not self.prefill_done:
            candidate_logits = torch.cat([new_token_logit, candidate_logits], dim=1)
            self.prefill_done = True

        return input_ids, candidate_logits

    def _reset(self) -> None:
        """Reset state for new generation session."""
        self.prefill_done = False
