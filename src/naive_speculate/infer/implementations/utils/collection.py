import torch


# TODO: find a more elegant way to support optional logits collection to save memory.
class OutputCollection:
    """Container for model intermediate outputs during decode or prefill.

    The user should call `update` method to collect outputs on demand during
    decode or prefill process, and call `finalize` method to get the final collected outputs.

    Attributes:
        output_ids (list[torch.Tensor]): Collected output ids, each of shape `[batch_size, 1]`.
        output_logits (list[torch.Tensor]): Collected output logits, each of shape `[batch_size, 1, vocab_size]`.
    """

    _output_ids: list[torch.Tensor]
    _output_logits: list[torch.Tensor]

    def __init__(self) -> None:
        self._output_ids = []
        self._output_logits = []

    def update(self, output_ids: torch.Tensor, output_logits: torch.Tensor) -> None:
        """Update collected outputs.

        Args:
            output_ids (torch.Tensor): New output ids to collect.
            output_logits (torch.Tensor): New output logits to collect.
        """
        self._output_ids.append(output_ids)
        self._output_logits.append(output_logits)

    def clear(self) -> None:
        """Clear collected outputs."""
        self._output_ids = []
        self._output_logits = []

    def find(self, token_id: int, start_idx: int) -> int:
        """Find the first occurrence of a token id in the collected output ids, starting from `start_idx`.

        Incur once device synchronization.

        Currently only reasonable for batch_size=1 scenarios.

        Args:
            token_id (int): The token id to search for.
            start_idx (int): The index to start searching from.

        Returns:
            int: The index of the first occurrence of the token id within the search length.
                -1 if not found.
        """
        if not self._output_ids or start_idx < 0:
            return -1

        tokens_to_search = self._output_ids[start_idx:]
        tokens_to_search = torch.cat(tokens_to_search, dim=1).squeeze(0).cpu()

        token_found, token_idx = torch.max((tokens_to_search == token_id), dim=0)

        if token_found.item():
            return start_idx + int(token_idx.item())

        return -1

    def finalize(self, num_tokens_trim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Finalize collected outputs and return them.

        Return empty tensors if no outputs have been collected or
        `num_tokens_trim` is greater than or equal to the number of collected tokens.

        If `num_tokens_trim <= 0`, return all collected outputs.

        Otherwise, trim the last `num_tokens_trim` tokens from the collected outputs.

        Args:
            num_tokens_trim (int): Number of tokens to trim from the end of the outputs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of collected output ids and logits.
        """
        output_ids = self._output_ids
        output_logits = self._output_logits

        if num_tokens_trim > 0:
            output_ids = output_ids[:-num_tokens_trim]
            output_logits = output_logits[:-num_tokens_trim]

        if not output_ids or not output_logits:
            return torch.empty(0), torch.empty(0)

        return torch.cat(output_ids, dim=1), torch.cat(output_logits, dim=1)
