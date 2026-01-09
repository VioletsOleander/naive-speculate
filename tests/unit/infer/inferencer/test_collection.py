from typing import NamedTuple, Protocol

import pytest
import torch

from naive_speculate.infer.inferencer.utils.collection import OutputCollection


class TensorShape(NamedTuple):
    batch_size: int
    num_tokens: int
    vocab_size: int


class CollectionFactory(Protocol):
    def __call__(self, num_collected_tokens: int) -> OutputCollection: ...


TENSOR_SHAPES = (
    TensorShape(batch_size=1, num_tokens=1, vocab_size=2),
    TensorShape(batch_size=1, num_tokens=1, vocab_size=4),
    TensorShape(batch_size=1, num_tokens=1, vocab_size=8),
)


@pytest.fixture(params=TENSOR_SHAPES)
def collection_factory(
    request: pytest.FixtureRequest,
) -> CollectionFactory:
    shape: TensorShape = request.param

    def _make_collection(num_collected_tokens: int) -> OutputCollection:
        collection = OutputCollection()

        token_id_gen = (torch.ones(shape[:-1]) * idx for idx in range(num_collected_tokens))
        token_logits_gen = (torch.ones(shape) for _ in range(num_collected_tokens))
        for token_id, token_logits in zip(token_id_gen, token_logits_gen, strict=True):
            collection.update(token_id, token_logits)

        return collection

    return _make_collection


class TestCollectionFind:
    class Param(NamedTuple):
        num_tokens: int
        token_id: int
        start_idx: int
        expected_idx: int

    PARAMS = (
        Param(num_tokens=5, token_id=3, start_idx=1, expected_idx=3),  # normal case
        Param(num_tokens=4, token_id=2, start_idx=3, expected_idx=-1),  # large start_idx
        Param(num_tokens=4, token_id=2, start_idx=4, expected_idx=-1),  # large start_idx
        Param(num_tokens=3, token_id=0, start_idx=-1, expected_idx=-1),  # negative start_idx
        Param(num_tokens=3, token_id=3, start_idx=0, expected_idx=-1),  # large token_id
    )

    @pytest.mark.parametrize("param", PARAMS)
    def test_collection_find(
        self,
        param: Param,
        collection_factory: CollectionFactory,
    ) -> None:
        collection = collection_factory(num_collected_tokens=param.num_tokens)
        result_idx = collection.find(token_id=param.token_id, start_idx=param.start_idx)

        assert result_idx == param.expected_idx


class TestCollectionFinalize:
    class Param(NamedTuple):
        num_tokens: int
        num_tokens_trim: int

    PARAMS = (
        Param(num_tokens=5, num_tokens_trim=2),  # normal case
        Param(num_tokens=3, num_tokens_trim=0),  # no trim case
        Param(num_tokens=3, num_tokens_trim=-1),  # no trim case
        Param(num_tokens=2, num_tokens_trim=5),  # all trim case
        Param(num_tokens=0, num_tokens_trim=0),  # empty case
    )

    @pytest.mark.parametrize("param", PARAMS)
    def test_collection_finalize(
        self,
        param: Param,
        collection_factory: CollectionFactory,
    ) -> None:
        collection = collection_factory(num_collected_tokens=param.num_tokens)

        # all trim case or empty case
        if param.num_tokens_trim >= param.num_tokens or param.num_tokens == 0:
            expected_ids = torch.empty(0)
            expected_logits = torch.empty(0)
        # no trim case
        elif param.num_tokens_trim <= 0:
            expected_ids = torch.cat(collection._output_ids, dim=-1)  # noqa: SLF001
            expected_logits = torch.cat(collection._output_logits, dim=-2)  # noqa: SLF001
        # normal case
        else:
            expected_ids = torch.cat(collection._output_ids[: -param.num_tokens_trim], dim=-1)  # noqa: SLF001
            expected_logits = torch.cat(collection._output_logits[: -param.num_tokens_trim], dim=-2)  # noqa: SLF001

        result_ids, result_logits = collection.finalize(
            num_tokens_trim=param.num_tokens_trim,
        )
        assert torch.equal(result_ids, expected_ids)
        assert torch.equal(result_logits, expected_logits)
