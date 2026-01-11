"""Provide factory functions to conduct creation of various components."""

from typing import TYPE_CHECKING

from naive_speculate.config.registry import InferencerType, KVCacheType, ModelFamily
from naive_speculate.speculate.drafter import Drafter
from naive_speculate.speculate.scorer import Scorer

if TYPE_CHECKING:
    from naive_speculate.infer import Inferencer, KVCache, LanguageModel

__all__ = ["make_drafter", "make_inferencer", "make_kvcache", "make_lm", "make_scorer"]


def make_lm(model_name: str) -> LanguageModel:
    family_name = model_name.split("/")[0].upper()
    match ModelFamily[family_name]:
        case ModelFamily.QWEN3:
            import naive_speculate.infer.lm.qwen3 as impl_module  # noqa: PLC0415

    lm_class = impl_module.LanguageModelImpl
    return lm_class(model_name=model_name)


def make_inferencer(language_model: LanguageModel, inferencer_type: InferencerType) -> Inferencer:
    match inferencer_type:
        case InferencerType.BASIC:
            from naive_speculate.infer.inferencer.basic import BasicInferencer  # noqa: PLC0415

            inferencer_class = BasicInferencer
        case InferencerType.CHUNKWISE:
            from naive_speculate.infer.inferencer.chunkwise import (  # noqa: PLC0415
                ChunkwiseDecodeInferencer,
            )

            inferencer_class = ChunkwiseDecodeInferencer

    return inferencer_class(language_model=language_model)


def make_drafter(inferencer: Inferencer) -> Drafter:
    return Drafter(inferencer=inferencer)


def make_scorer(language_model: LanguageModel) -> Scorer:
    return Scorer(language_model=language_model)


def make_kvcache(kvcache_type: KVCacheType) -> KVCache:
    match kvcache_type:
        case KVCacheType.DYNAMIC:
            import naive_speculate.infer.kvcache.dynamic as impl_module  # noqa: PLC0415
        case KVCacheType.DYNAMIC_NO_UPDATE:
            import naive_speculate.infer.kvcache.dynamic_no_update as impl_module  # noqa: PLC0415

    kvcache_class = impl_module.KVCacheImpl
    return kvcache_class()
