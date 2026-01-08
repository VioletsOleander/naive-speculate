from typing import TYPE_CHECKING, override

import torch

from naive_speculate.score import Scorer as ScorerProtocol
from naive_speculate.score import ScoreResult

if TYPE_CHECKING:
    from naive_speculate.infer import Inferencer, KVCache


class Scorer(ScorerProtocol):
    """Implements `Scorer` protocol.

    Delegates scoring to an `Inferencer` instance.

    Refers to the protocol `Scorer` for more details.

    Attributes:
        inferencer (Inferencer): The inferencer used for scoring.
    """

    inferencer: Inferencer

    def __init__(self, inferencer: Inferencer) -> None:
        self.inferencer = inferencer

    @torch.no_grad()
    @override
    def score(
        self,
        query_token_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> ScoreResult:
        token_logits = self.inferencer.forward(query_token_ids=query_token_ids, kv_cache=kv_cache)
        return ScoreResult(token_logits=token_logits)


ScorerImpl = Scorer
