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
            import naive_speculate.infer.impl.inferencer.concrete.qwen3 as impl_module  # noqa: PLC0415

    model_class = impl_module.Model
    model = model_class(model_name=model_name)

    # 2. make inferencer
    match inferencer_type:
        case InferencerType.BASIC:
            inferencer_class = impl_module.BasicInferencerImpl
        case InferencerType.CHUNKWISE:
            inferencer_class = impl_module.ChunkwiseInferencerImpl

    return inferencer_class(model=model)


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
            import naive_speculate.infer.impl.kvcache.dynamic_cache as impl_module  # noqa: PLC0415
        case KVCacheType.DYNAMIC_NO_UPDATE:
            import naive_speculate.infer.impl.kvcache.dynamic_no_update_cache as impl_module  # noqa: PLC0415

    kvcache_class = impl_module.KVCacheImpl
    return kvcache_class()
