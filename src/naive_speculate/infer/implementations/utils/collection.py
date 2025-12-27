import torch


# TODO: find a more elegant way to support optional logits collection to save memory.
class OutputCollection:
    """Container for model intermediate outputs during decode or prefill.

    The user should call `update` method to collect outputs on demand during
    decode or prefill process, and call `finalize` method to get the final collected outputs.

    Attributes:
        output_ids (list[torch.Tensor]): Collected output ids.
        output_logits (list[torch.Tensor]): Collected output logits.
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

    def rfind(self, token_id: int) -> int:
        """Find the last occurrence of a token id in the collected output ids.

        Args:
            token_id (int): The token id to search for.

        Returns:
            int: The index of the last occurrence of the token id, or -1 if not found.
        """
        if not self._output_ids:
            return -1

        concatenated_ids = torch.cat(self._output_ids, dim=1)
        flattened_ids = concatenated_ids.view(-1).tolist()

        for index in range(len(flattened_ids) - 1, -1, -1):
            if flattened_ids[index] == token_id:
                return index

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
