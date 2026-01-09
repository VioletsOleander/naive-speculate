from typing import TYPE_CHECKING, override

from naive_speculate.infer.impl.inferencer.abstract.basic import BasicInferencer
from naive_speculate.infer.impl.inferencer.abstract.chunkwise import ChunkwiseDecodeInferencer

if TYPE_CHECKING:
    import torch

    from naive_speculate.infer import KVCache

    from .fake_model import FakeModel


class FakeBasicInferencer(BasicInferencer):
    fake_model: FakeModel

    def __init__(self, fake_model: FakeModel) -> None:
        self.fake_model = fake_model

    @override
    def _eos_token_id(self) -> int:
        return self.fake_model.eos_token_id

    @override
    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        return self.fake_model.forward(query_token_ids, kv_cache)


class FakeChunkwiseInferencer(ChunkwiseDecodeInferencer):
    fake_model: FakeModel
    _decode_chunk_size: int = 8

    def __init__(self, fake_model: FakeModel) -> None:
        self.fake_model = fake_model

    @property
    @override
    def decode_chunk_size(self) -> int:
        return self._decode_chunk_size

    @override
    def _eos_token_id(self) -> int:
        return self.fake_model.eos_token_id

    @override
    def forward(self, query_token_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        return self.fake_model.forward(query_token_ids, kv_cache)
