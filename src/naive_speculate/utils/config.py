import tomllib
from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel


class SampleStrategy(StrEnum):
    """Sampling strategies for token generation.

    Attributes:
        RANDOM: Sample tokens probabilistically according to the token distribution over vocabulary.
        GREEDY: Always select the token with the highest probability (argmax).
    """

    RANDOM = auto()
    GREEDY = auto()


class VerifyStrategy(StrEnum):
    """Verification strategies for speculative decoding.

    Attributes:
        GREEDY_MATCH: Verify drafted tokens using greedy matching.
        SPECULATIVE_SAMPLING: Verify drafted tokens using speculative sampling.
    """

    GREEDY_MATCH = auto()
    SPECULATIVE_SAMPLING = auto()


class DraftConfig(BaseModel):
    """Configuration related to the drafting process in speculative decoding.

    Attributes:
        model_name (str): Name of the underlying `transformers` model used for drafting.
            This name will be used to load the model and tokenizer from `transformers` library.
            Example: `"Qwen3/Qwen3-0.6B"`
        sample_strategy (SampleStrategy): Sampling strategy for token drafting.
            Options: `("random", "greedy")`, case sensitive.
            Default to `"random"`.
        num_draft_tokens (int): Number of tokens to draft in each speculation step.
            Must be a positive integer.
            Default to `5`.
    """

    model_name: str
    sample_strategy: SampleStrategy = SampleStrategy.RANDOM
    num_draft_tokens: int = 5


class VerifyConfig(BaseModel):
    """Configuration related to the verification process in speculative decoding.

    Attributes:
        model_name (str): Name of the underlying `transformers` model used for verification.
            This name will be used to load the model and tokenizer from `transformers` library.
            Example: `"Qwen3/Qwen3-8B"`
        verify_strategy (VerifyStrategy): Verification strategy for drafted tokens.
            Options: `("speculative_sampling", "greedy_match")`, case sensitive.
            Default to `"speculative_sampling"`.
    """

    model_name: str
    verify_strategy: VerifyStrategy = VerifyStrategy.SPECULATIVE_SAMPLING


class SpeculateConfig(BaseModel):
    """Configuration for the speculative decoding process.

    Refers to the docstring of `DraftConfig` and `VerifyConfig` for more details.

    Attributes:
        draft (DraftConfig): Configuration for the drafting process.
        verify (VerifyConfig): Configuration for the verification process.
    """

    draft: DraftConfig
    verify: VerifyConfig

    @classmethod
    def from_toml(cls, toml_path: str) -> SpeculateConfig:
        """Load configuration from a TOML file.

        Args:
            toml_path (str): Path to the TOML configuration file.

        Returns:
            SpeculateConfig: An instance of SpeculateConfig populated with values from the TOML file.
        """
        with Path(toml_path).open("rb") as f:
            config_dict = tomllib.load(f)

        return cls(**config_dict)
