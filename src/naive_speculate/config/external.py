"""User facing interface, defining external user specifiable configuration options."""

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from .registry import ModelFamily
from .strategy import SampleStrategy, VerifyStrategy


def _is_supported_model_family(model_name: str) -> str:
    """Check if the given model name belongs to a supported model family."""
    family_name = model_name.split("/")[0].upper()
    if family_name not in ModelFamily.__members__:
        raise ValueError(
            f"Model family '{family_name.capitalize()}' is not supported. "
            f"Supported families are: {[e.name.capitalize() for e in ModelFamily]}."
        )
    return model_name


class UserDraftConfig(BaseModel):
    """User specifiable configuration related to the drafting process in speculative decoding.

    Attributes:
        model_name (str): Name of the underlying `transformers` model used for drafting.
            This name will be used to load the model and tokenizer from `transformers` library.
            Example: `"Qwen/Qwen3-0.6B"`
        sample_strategy (SampleStrategy): Sampling strategy for token drafting.
            Default to `SampleStrategy.RANDOM`.
        num_draft_tokens (int): Number of tokens to draft in each speculation step.
            Must be a positive integer.
            Default to `5`.
    """

    model_name: Annotated[str, AfterValidator(_is_supported_model_family)]
    sample_strategy: SampleStrategy = SampleStrategy.RANDOM
    num_draft_tokens: int = Field(default=5, gt=0)


class UserVerifyConfig(BaseModel):
    """User specifiable configuration related to the verification process in speculative decoding.

    Attributes:
        model_name (str): Name of the underlying `transformers` model used for verification.
            This name will be used to load the model and tokenizer from `transformers` library.
            Example: `"Qwen/Qwen3-8B"`
        verify_strategy (VerifyStrategy): Verification strategy for drafted tokens.
            Default to `VerifyStrategy.SPECULATIVE_SAMPLING`.
    """

    model_name: Annotated[str, AfterValidator(_is_supported_model_family)]
    verify_strategy: VerifyStrategy = VerifyStrategy.SPECULATIVE_SAMPLING


class UserSpeculateConfig(BaseModel):
    """User specifiable configuration for the speculative decoding process.

    Refers to the docstring of `DraftConfig` and `VerifyConfig` for more details.

    Attributes:
        draft (DraftConfig): Configuration for the drafting process.
        verify (VerifyConfig): Configuration for the verification process.
    """

    draft: UserDraftConfig
    verify: UserVerifyConfig

    @classmethod
    def from_toml(cls, toml_path: str) -> UserSpeculateConfig:
        """Load configuration from a TOML file.

        Args:
            toml_path (str): Path to the TOML configuration file.

        Returns:
            UserSpeculateConfig: Config instance populated with values from the TOML file.
        """
        with Path(toml_path).open("rb") as f:
            config_dict = tomllib.load(f)

        return cls(**config_dict)
