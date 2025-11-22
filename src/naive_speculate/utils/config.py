import tomllib
from dataclasses import dataclass
from pathlib import Path

from idna import decode


@dataclass
class Config:
    drafter_model_name: str = ""
    verifier_model_name: str = ""
    max_new_tokens: int = 0
    decode_method: str = ""

    @staticmethod
    def from_file(config_path: str) -> Config:
        with open(Path(config_path), "rb") as f:
            config_dict = tomllib.load(f)

        config_dict = {
            "drafter_model_name": config_dict["draft"]["model_name"],
            "verifier_model_name": config_dict["verify"]["model_name"],
            "max_new_tokens": config_dict["max_new_tokens"],
            "decode_method": config_dict["verify"]["decode_method"],
        }

        config = Config(**config_dict)
        config.validate_self()
        return config

    def validate_self(self):
        """Validate the configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.drafter_model_name == "" or self.verifier_model_name == "":
            raise ValueError("Model names must be specified in the config.")

        if self.decode_method not in ["greedy", "stochastic"]:
            raise ValueError("Decode method must be either 'greedy' or 'stochastic'.")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")
