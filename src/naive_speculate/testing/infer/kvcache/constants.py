from typing import NamedTuple

import torch

from naive_speculate.infer import KVState


class _KVCacheShape(NamedTuple):
    batch_size: int
    num_attention_heads: int
    num_tokens: int
    head_dim: int


_KVCACHE_SHAPES = [
    _KVCacheShape(batch_size=1, num_attention_heads=2, num_tokens=16, head_dim=8),
    _KVCacheShape(batch_size=2, num_attention_heads=2, num_tokens=8, head_dim=8),
    _KVCacheShape(batch_size=4, num_attention_heads=1, num_tokens=4, head_dim=8),
]

_NUM_LAYERS = [1, 2, 4]

KVSTATES = [
    [
        KVState(
            keys=torch.empty(shape),
            values=torch.empty(shape),
        )
        for shape in _KVCACHE_SHAPES
    ]
    for _ in _NUM_LAYERS
]

NUM_TOKENS_CROP = [0, 2, 4]
