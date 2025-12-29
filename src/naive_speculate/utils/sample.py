from enum import StrEnum

import torch


class SampleStrategy(StrEnum):
    """Sampling strategies for token generation.

    Attributes:
        RANDOM = "random": Sample tokens probabilistically according to
            the softmax distribution over `token_logits`.
        GREEDY = "greedy": Always select the token with
            the highest probability (argmax) from `token_logits`.
    """

    RANDOM = "random"
    GREEDY = "greedy"


def sample_tokens(token_logits: torch.Tensor, sampling_strategy: SampleStrategy) -> torch.Tensor:
    """Sample token ids from `token_logits` according to `sampling_strategy`.

    Args:
        token_logits (torch.Tensor): Logits of shape `[batch_size, vocab_size]`.
        sampling_strategy (SampleStrategy): Sampling strategy to use.

    Returns:
        torch.Tensor: Sampled next token ids of shape `[batch_size, 1]`.

    Raises:
        ValueError: If `sampling_strategy` is unknown.
    """
    match sampling_strategy:
        case SampleStrategy.GREEDY:
            token_ids = torch.argmax(token_logits, dim=-1, keepdim=True)
        case SampleStrategy.RANDOM:
            probs = torch.softmax(token_logits, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1)
        case _:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    return token_ids
