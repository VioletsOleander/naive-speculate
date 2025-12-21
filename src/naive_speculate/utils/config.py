import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpeculateConfig:
    drafter_model_name: str = ""
    verifier_model_name: str = ""
    max_new_tokens: int = 0
    decode_method: str = ""
    draft_tokens_num: int = 0

    @staticmethod
    def from_dict(config_dict: dict) -> SpeculateConfig:
        config = SpeculateConfig(**config_dict)
        config.validate_self()
        return config

    @staticmethod
    def from_file(config_path: str) -> SpeculateConfig:
        with open(Path(config_path), "rb") as f:
            config_dict = tomllib.load(f)

        config_dict = {
            "drafter_model_name": config_dict["draft"]["model_name"],
            "verifier_model_name": config_dict["verify"]["model_name"],
            "draft_tokens_num": config_dict["draft"]["draft_tokens_num"],
            "decode_method": config_dict["decode_method"],
            "max_new_tokens": config_dict["max_new_tokens"],
        }

        return SpeculateConfig.from_dict(config_dict)

    def validate_self(self) -> None:
        """Validate the configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.drafter_model_name == "" or self.verifier_model_name == "":
            raise ValueError("Model names must be specified in the config.")

        if self.decode_method not in ["greedy", "random"]:
            raise ValueError("Decode method must be either 'greedy' or 'random'.")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")

        if self.draft_tokens_num <= 0:
            raise ValueError("draft_tokens_num must be a positive integer.")
