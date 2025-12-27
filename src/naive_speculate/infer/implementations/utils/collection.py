import torch


# TODO: find a more elegant way to support optional logits collection to save memory.
class OutputCollection:
    """Container for model intermediate outputs during decode or prefill.

    The user should call `update` method to collect outputs on demand during
    decode or prefill process, and call `finalize` method to get the final collected outputs.

    Attributes:
        output_ids (torch.Tensor): Collected output ids.
        output_logits (list[torch.Tensor]): Collected output logits.
    """

    output_ids: list[torch.Tensor]
    output_logits: list[torch.Tensor]

    def __init__(self) -> None:
        self.output_ids = []
        self.output_logits = []

    def update(self, output_ids: torch.Tensor, output_logits: torch.Tensor) -> None:
        """Update collected outputs.

        Args:
            output_ids (torch.Tensor): New output ids to collect.
            output_logits (torch.Tensor | None): New output logits to collect.

        Raises:
            ValueError: If `self.collect_logits` is True but `output_logits` is None.
        """
        self.output_ids.append(output_ids)
        self.output_logits.append(output_logits)

    def clear(self) -> None:
        """Clear collected outputs."""
        self.output_ids = []
        self.output_logits = []

    def rfind(self, token_id: int) -> int:
        """Find the last occurrence of a token id in the collected output ids.

        Args:
            token_id (int): The token id to search for.

        Returns:
            int: The index of the last occurrence of the token id, or -1 if not found.
        """
        if not self.output_ids:
            return -1

        concatenated_ids = torch.cat(self.output_ids, dim=1)
        flattened_ids = concatenated_ids.view(-1).tolist()

        for index in range(len(flattened_ids) - 1, -1, -1):
            if flattened_ids[index] == token_id:
                return index

        return -1

    def finalize(self, num_tokens_trim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Finalize collected outputs and return them.

        If no outputs have been collected, return empty tensors.

        If `num_tokens_trim` > 0, trim that many tokens from the end of the outputs,
        otherwise, return all collected outputs.

        Args:
            output_type (OutputType): The type of output to return, either "prefill" or "decode".
            num_tokens_trim (int): Number of tokens to trim from the end of the outputs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of collected output ids and logits.
        """
        if not self.output_ids or not self.output_logits:
            return torch.empty(0), torch.empty(0)

        output_ids = self.output_ids
        output_logits = self.output_logits
        if num_tokens_trim > 0:
            output_ids = output_ids[:-num_tokens_trim]
            output_logits = output_logits[:-num_tokens_trim]

        return (torch.cat(output_ids, dim=1), torch.cat(output_logits, dim=1))
