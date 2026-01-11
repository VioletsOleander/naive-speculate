"""Provide `load_config`."""

from .external import UserSpeculateConfig
from .internal import DraftConfig, InferenceConfig, SpeculateConfig, VerifyConfig
from .registry import InferencerType, KVCacheType

__all__ = ["load_config"]


def load_config(config_path: str) -> SpeculateConfig:
    """Load user specified configuration from file into `SpeculateConfig` for internal use.

    Args:
        config_path (str): Path to the user specified configuration file.

    Returns:
        SpeculateConfig: Specific configuration for internal use.
    """
    user_config = UserSpeculateConfig.from_toml(config_path)
    return _internalize(user_config)


def _internalize(user_config: UserSpeculateConfig) -> SpeculateConfig:
    """Translate external user specified configuration into `SpeculateConfig`.

    Args:
        user_config (UserSpeculateConfig): User specified configuration.

    Returns:
        SpeculateConfig: Specific configuration for internal use.
    """

    def make_infer_config(model_name: str) -> InferenceConfig:
        return InferenceConfig(
            model_name=model_name,
            kvcache_type=KVCacheType.DYNAMIC_NO_UPDATE,
            inferencer_type=InferencerType.CHUNKWISE,
        )

    draft_config = DraftConfig(
        sample_strategy=user_config.draft.sample_strategy,
        num_draft_tokens=user_config.draft.num_draft_tokens,
        infer=make_infer_config(user_config.draft.model_name),
    )

    verify_config = VerifyConfig(
        verify_strategy=user_config.verify.verify_strategy,
        infer=make_infer_config(user_config.verify.model_name),
    )

    return SpeculateConfig(
        draft=draft_config,
        verify=verify_config,
    )
