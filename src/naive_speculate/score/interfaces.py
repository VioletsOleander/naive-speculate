from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    import torch

    from naive_speculate.infer import KVCache


class ScoreResult(NamedTuple):
    """Output of `Scorer.score` method.

    Contains the token logits for the given query tokens (except for the first ones),
    and the token logits for the next possible tokens.

    Attributes:
        token_logits (torch.Tensor): Logits for the query tokens. Shape `[batch_size, num_query_tokens, vocab_size]`.
    """

    token_logits: torch.Tensor


class Scorer(Protocol):
    """Scorer is able to process given tokens and produce corresponding token logits (scores).

    Scorer can either be itself a language model, or delegate the scoring to an underlying
    language model.

    Scorer only scores the given query tokens, and does not generate any new tokens even though
    the scores can be used to do so.

    In the context of speculative decoding, scoring is part of the verification process, where
    the speculative decoder will do speculative sampling based on the token logits produced
    by the scorer.
    """

    def score(
        self,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> ScoreResult:
        """Score the given query tokens using the provided key-value cache.

        `kv_cache` will be updated internally as a side effect of this method.

        Return the token logits corresponding to the query tokens (except for the first ones),
        and the token logits corresponding to the next possible tokens.

        Args:
            query_token_ids (torch.Tensor): Query tokens to be scored. Shape `[batch_size, num_query_tokens]`.
            kv_cache (torch.Tensor): Keys and values tensor for past context tokens.

        Returns:
            ScoreResult: The scoring result containing token logits.
                Shape `[batch_size, num_query_tokens, vocab_size]`.
        """
        ...
