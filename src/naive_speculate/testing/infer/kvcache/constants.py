from itertools import product
from typing import TYPE_CHECKING, NamedTuple

import torch

from naive_speculate.infer import KVState

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["CROP_RATIOS", "KVSTATES"]

CROP_RATIOS = (
    -1.0,  # special value: crop one
    0.0,  # no crop
    0.25,
    0.5,
    1.0,  # crop all
)


class _KVCacheShape(NamedTuple):
    batch_size: int
    num_attention_heads: int
    num_tokens: int
    head_dim: int


_KVCACHE_SHAPES = (
    _KVCacheShape(batch_size=1, num_attention_heads=2, num_tokens=16, head_dim=8),
    _KVCacheShape(batch_size=2, num_attention_heads=2, num_tokens=8, head_dim=8),
    _KVCacheShape(batch_size=4, num_attention_heads=1, num_tokens=4, head_dim=8),
)

_NUM_LAYERS = (1, 2, 4)


def _gen_kvstates() -> Generator[list[KVState]]:
    for num_layer, shape in product(_NUM_LAYERS, _KVCACHE_SHAPES):
        yield [
            KVState(keys=torch.empty(shape), values=torch.empty(shape)) for _ in range(num_layer)
        ]


KVSTATES = _gen_kvstates()
