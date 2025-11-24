from re import M
from sre_parse import Tokenizer
from typing import TypeAlias

from torch import Tensor
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    Qwen2TokenizerFast,
    Qwen3ForCausalLM,
)
from transformers.generation.utils import GenerateEncoderDecoderOutput

from naive_speculate.modules.verify import ModelOutputType
from naive_speculate.utils.config import Config

ModelType: TypeAlias = Qwen3ForCausalLM
TokenizerType: TypeAlias = Qwen2TokenizerFast
ModelOutputType: TypeAlias = GenerateEncoderDecoderOutput


class Drafter:
    config: Config
    model: ModelType
    tokenizer: TokenizerType

    def __init__(self, config: Config):
        self.config = config

        self.model = Qwen3ForCausalLM.from_pretrained(
            config.drafter_model_name, device_map="auto", dtype="auto"
        )
        assert self.model.generation_config is not None
        self.model.generation_config.return_dict_in_generate = True
        self.model.generation_config.output_scores = True
        self.model.generation_config.max_new_tokens = 1000

        self.tokenizer = AutoTokenizer.from_pretrained(config.drafter_model_name)
        # self.tokenizer = Qwen2TokenizerFast.from_pretrained(config.drafter_model_name)
        self.tokenizer.padding_side = "left"

    def draft(self, model_input: BatchEncoding) -> GenerateEncoderDecoderOutput:
        draft: GenerateEncoderDecoderOutput = self.model.generate(**model_input)  # type: ignore
        return draft

    def tokenize(self, input_texts: list[str]) -> BatchEncoding:
        return self.tokenizer(input_texts, return_tensors="pt", padding=True).to(
            self.model.device
        )

    def detokenize(self, token_ids: list[int] | Tensor) -> list[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def __str__(self) -> str:
        string = f"Drafter(model_name={self.config.drafter_model_name})\nType: {type(self.model)}\n{self.model}"
        return string
