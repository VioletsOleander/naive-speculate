import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpeculateConfig:
    """Configuration for the speculative generation process.

    Attributes:
        drafter_model_name (str): Name of the drafter model.
        verifier_model_name (str): Name of the verifier model.
        max_new_tokens (int): Maximum number of new tokens to generate.
        decode_method (str): Decoding method to use ('greedy' or 'random').
        draft_tokens_num (int): Number of tokens to draft in each speculation step.
        streaming (bool): Whether to enable streaming output.
    """

    drafter_model_name: str = ""
    verifier_model_name: str = ""
    max_new_tokens: int = 0
    decode_method: str = ""
    draft_tokens_num: int = 0
    streaming: bool = False

    @staticmethod
    def from_dict(config_dict: dict) -> SpeculateConfig:
        config = SpeculateConfig(**config_dict)
        config.validate_self()
        return config

    @staticmethod
    def from_file(config_path: str) -> SpeculateConfig:
        with open(Path(config_path), "rb") as f:
            config_dict = tomllib.load(f)

        general_configs = config_dict.get("general", {})

        config_dict = {
            "decode_method": general_configs.get("decode_method", "greedy"),
            "max_new_tokens": general_configs.get("max_new_tokens", 1024),
            "streaming": general_configs.get("streaming", False),
            "drafter_model_name": config_dict["draft"]["model_name"],
            "draft_tokens_num": config_dict["draft"].get("draft_tokens_num", 5),
            "verifier_model_name": config_dict["verify"]["model_name"],
        }

        return SpeculateConfig.from_dict(config_dict)

    def validate_self(self) -> None:
        """Validate the configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.decode_method not in ["greedy", "random"]:
            raise ValueError("Decode method must be either 'greedy' or 'random'.")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")

        if not isinstance(self.streaming, bool):
            raise ValueError("streaming must be a boolean value.")

        if self.drafter_model_name == "" or self.verifier_model_name == "":
            raise ValueError("Model names must be specified in the config.")

        if self.draft_tokens_num <= 0:
            raise ValueError("draft_tokens_num must be a positive integer.")
