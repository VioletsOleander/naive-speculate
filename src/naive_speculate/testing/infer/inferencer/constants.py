from typing import NamedTuple

from naive_speculate.testing.infer.lm.fake import FakeLMConfig

__all__ = ["EOS_POSITIONS", "FAKE_LM_CONFIGS", "MAX_NEW_TOKENS", "QUERY_SHAPES", "QueryShape"]

FAKE_LM_CONFIGS = (
    FakeLMConfig(eos_token_id=0, vocab_size=5, num_layers=1, num_heads=2, embed_dim=4),
    FakeLMConfig(eos_token_id=1, vocab_size=10, num_layers=2, num_heads=4, embed_dim=8),
    FakeLMConfig(eos_token_id=2, vocab_size=15, num_layers=4, num_heads=6, embed_dim=12),
)


class QueryShape(NamedTuple):
    batch_size: int
    num_query_tokens: int


QUERY_SHAPES = (
    QueryShape(batch_size=1, num_query_tokens=15),
    QueryShape(batch_size=1, num_query_tokens=10),
    QueryShape(batch_size=1, num_query_tokens=5),
)

MAX_NEW_TOKENS = (1, 5, 10)
EOS_POSITIONS = (1, 5, 10)
