"""Define `Scorer`, implementing token scoring (logits computing) functionality."""

from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from naive_speculate.infer import KVCache, LanguageModel

__all__ = ["ScoreResult", "Scorer"]


class ScoreResult(NamedTuple):
    """Output of `Scorer.score` method.

    Contains the logits at each position of the query tokens. The logits at position `i`
    are used to predict the token at position `i+1`.

    Attributes:
        token_logits (torch.Tensor):  Logits at the query token positions.
            Shape `[batch_size, num_query_tokens, vocab_size]`.
    """

    token_logits: torch.Tensor


class Scorer:
    """Scorer is able to process given tokens and produce corresponding token logits (scores).

    Scorer delegates scoring to an `LanguageModel` instance.

    Scorer only scores the given query tokens, and does not generate any new tokens even though
    the scores can be used to do so.

    In the context of speculative decoding, scoring is part of the verification process, where
    the speculative decoder will do speculative sampling based on the token logits produced
    by the scorer.

    Attributes:
        language_model (LanguageModel): The language model used for scoring.
    """

    language_model: LanguageModel

    def __init__(self, language_model: LanguageModel) -> None:
        self.language_model = language_model

    @torch.no_grad()
    def score(
        self,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> ScoreResult:
        """Score the given query tokens using the provided key-value cache.

        `kv_cache` will be updated internally as a side effect of this method.

        Return the output includes logits for all query token positions,
        where position `i` gives the logits for predicting token `i+1`.

        Args:
            query_token_ids (torch.Tensor): Query tokens to be scored. Shape `[batch_size, num_query_tokens]`.
            kv_cache (KVCache): Keys and values tensor for past context tokens.

        Returns:
            ScoreResult: The scoring result containing token logits.
                Shape `[batch_size, num_query_tokens, vocab_size]`.
        """
        token_logits = self.language_model.forward(
            query_token_ids=query_token_ids, kv_cache=kv_cache
        )
        return ScoreResult(token_logits=token_logits)
