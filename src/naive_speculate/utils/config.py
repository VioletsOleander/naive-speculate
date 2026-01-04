import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class SampleStrategy(StrEnum):
    """Sampling strategies for token generation.

    Attributes:
        RANDOM = "random": Sample tokens probabilistically according to
            the softmax distribution over `token_logits`.
        GREEDY = "greedy": Always select the token with
            the highest probability (argmax) from `token_logits`.
    """

    RANDOM = "random"
    GREEDY = "greedy"


class VerifyStrategy(StrEnum):
    """Verification strategies for speculative decoding.

    Attributes:
        GREEDY_MATCH: Verify drafted tokens using greedy matching.
        SPECULATIVE_SAMPLE: Verify drafted tokens using speculative sampling.
    """

    GREEDY_MATCH = "greedy_match"
    SPECULATIVE_SAMPLE = "speculative_sample"


# TODO: migrate to pydantic for validation
@dataclass
class SpeculateConfig:
    """Configuration for the speculative generation process.

    Attributes:
        drafter_model_name (str): Used for loading the underlying `transformers` model for drafting.
        verifier_model_name (str): Used for loading the underlying `transformers` model for verification.
        sample_strategy (SampleStrategy): Sampling strategy for token drafting.
        verify_strategy (VerifyStrategy): Verification strategy for drafted tokens.
        num_draft_tokens (int): Number of tokens to draft in each speculation step.
        streaming (bool): Whether to enable streaming output.
    """

    drafter_model_name: str = ""
    verifier_model_name: str = ""
    sample_strategy: SampleStrategy = SampleStrategy.GREEDY
    verify_strategy: VerifyStrategy = VerifyStrategy.GREEDY_MATCH
    num_draft_tokens: int = 0
    streaming: bool = False

    @staticmethod
    def from_dict(config_dict: dict) -> SpeculateConfig:
        config = SpeculateConfig(**config_dict)
        config.validate_self()
        return config

    @staticmethod
    def from_file(config_path: str) -> SpeculateConfig:
        with Path(config_path).open("rb") as f:
            config_dict = tomllib.load(f)

        general_configs = config_dict.get("general", {})

        try:
            config_dict = {
                "decode_method": general_configs.get("decode_method", SampleStrategy.GREEDY),
                "verify_method": general_configs.get(
                    "verify_method", VerifyStrategy.SPECULATIVE_SAMPLE
                ),
                "max_new_tokens": general_configs.get("max_new_tokens", 1024),
                "streaming": general_configs.get("streaming", False),
                "drafter_model_name": config_dict["draft"]["model_name"],
                "num_draft_tokens": config_dict["draft"].get("num_draft_tokens", 5),
                "verifier_model_name": config_dict["verify"]["model_name"],
            }
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}") from e

        return SpeculateConfig.from_dict(config_dict)

    def validate_self(self) -> None:
        """Validate the configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.drafter_model_name == "" or self.verifier_model_name == "":
            raise ValueError("Model names must be specified in the config.")

        if self.num_draft_tokens <= 0:
            raise ValueError("num_draft_tokens must be a positive integer.")
