from typing import TYPE_CHECKING

from naive_speculate.config.registry import InferencerType, KVCacheType, ModelFamily

if TYPE_CHECKING:
    from naive_speculate.draft import Drafter
    from naive_speculate.infer import Inferencer, KVCache
    from naive_speculate.score import Scorer


def make_inferencer(model_name: str, inferencer_type: InferencerType) -> Inferencer:
    # 1. make model
    family_name = model_name.split("/")[0].upper()
    match ModelFamily[family_name]:
        case ModelFamily.QWEN3:
            import naive_speculate.infer.inferencer.model.qwen3 as impl_module  # noqa: PLC0415

    lm_class = impl_module.LanguageModelImpl
    language_model = lm_class(model_name=model_name)

    # 2. make inferencer
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
    import naive_speculate.draft.impl.drafter as impl_module  # noqa: PLC0415

    drafter_class = impl_module.DrafterImpl
    return drafter_class(inferencer=inferencer)


def make_scorer(inferencer: Inferencer) -> Scorer:
    import naive_speculate.score.impl.scorer as impl_module  # noqa: PLC0415

    scorer_class = impl_module.ScorerImpl
    return scorer_class(inferencer=inferencer)


def make_kvcache(kvcache_type: KVCacheType) -> KVCache:
    match kvcache_type:
        case KVCacheType.DYNAMIC:
            import naive_speculate.infer.kvcache.dynamic as impl_module  # noqa: PLC0415
        case KVCacheType.DYNAMIC_NO_UPDATE:
            import naive_speculate.infer.kvcache.dynamic_no_update as impl_module  # noqa: PLC0415

    kvcache_class = impl_module.KVCacheImpl
    return kvcache_class()
