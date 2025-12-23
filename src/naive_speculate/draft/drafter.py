from logging import Logger

import torch

from naive_speculate.models import QwenModel
from naive_speculate.utils import SpeculateConfig, logger_or_dummy


class Drafter(QwenModel):
    """Drafter model for generating candidate tokens.

    Attributes:
        num_draft_tokens (int): Number of tokens to draft in each round.
        decode_method (str): Should be 'greedy' or 'random'.
        logger (Logger)
    """

    num_draft_tokens: int
    decode_method: str
    logger: Logger

    def __init__(self, config: SpeculateConfig, logger: Logger | None = None):
        """Initialize the Drafter model.

        Args:
            config (SpeculateConfig)
            logger (Logger | None): If None, a dummy logger will be used, which ignores all logging calls.
        """
        super().__init__(config.drafter_model_name)
        self.num_draft_tokens = config.num_draft_tokens
        self.decode_method = config.decode_method
        self.logger = logger_or_dummy(logger)

    @torch.no_grad()
    def draft(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate candidate tokens given the context.

        Returns the updated tokens IDs and logits of newly generated tokens.

        Args:
            input_ids (torch.Tensor): The context's token IDs. Shape: `[batch_size, seq_len]`

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The updated token IDs and logits of the generated draft tokens.
            Shapes:
              - updated_input_ids: `[batch_size, seq_len + num_draft_tokens]`
              - candidate_logits: `[batch_size, num_draft_tokens, vocab_size]`

        Raises:
            ValueError: If `self.decode_method` is not supported.
        """
        num_draft_tokens = self.num_draft_tokens

        if not self.prefill_done:
            input_ids, candidate_logits = self._prefill(
                input_ids=input_ids,
                decode_method=self.decode_method,
                output_logits=True,
            )
            new_token_logits = candidate_logits[:, -1:, :]
            num_draft_tokens -= 1

        input_ids, candidate_logits = self._decode(
            input_ids=input_ids,
            max_new_tokens=num_draft_tokens,
            decode_method=self.decode_method,
            output_logits=True,
        )
        if not self.prefill_done:
            candidate_logits = torch.cat((new_token_logits, candidate_logits), dim=1)
            self.prefill_done = True

        return input_ids, candidate_logits
