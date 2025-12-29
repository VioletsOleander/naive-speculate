import tomllib
from dataclasses import dataclass
from pathlib import Path

from .sample import SampleStrategy


# TODO: migrate to pydantic for validation
@dataclass
class SpeculateConfig:
    """Configuration for the speculative generation process.

    Attributes:
        drafter_model_name (str): Name of the drafter model.
        verifier_model_name (str): Name of the verifier model.
        max_new_tokens (int): Maximum number of new tokens to generate.
        sample_strategy (SampleStrategy): Sampling strategy for generation.
        num_draft_tokens (int): Number of tokens to draft in each speculation step.
        streaming (bool): Whether to enable streaming output.
    """

    drafter_model_name: str = ""
    verifier_model_name: str = ""
    max_new_tokens: int = 0
    num_draft_tokens: int = 0
    sample_strategy: SampleStrategy = SampleStrategy.GREEDY
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
                "decode_method": general_configs.get("decode_method", "greedy"),
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
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")

        if self.drafter_model_name == "" or self.verifier_model_name == "":
            raise ValueError("Model names must be specified in the config.")

        if self.num_draft_tokens <= 0:
            raise ValueError("num_draft_tokens must be a positive integer.")
