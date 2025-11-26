from typing import TypeAlias

import torch
import transformers
from torch import Tensor
from transformers import BatchEncoding

from naive_speculate.utility.config import Config

ModelType: TypeAlias = transformers.Qwen3ForCausalLM
TokenizerType: TypeAlias = transformers.Qwen2TokenizerFast
ModelOutputType: TypeAlias = transformers.generation.utils.GenerateDecoderOnlyOutput


class Drafter:
    config: Config
    model: ModelType
    tokenizer: TokenizerType

    def __init__(self, config: Config):
        self.config = config

        self.model = ModelType.from_pretrained(
            config.drafter_model_name, device_map="auto", dtype="auto"
        )
        self._prepare_generation_config()

        self.tokenizer = TokenizerType.from_pretrained(config.drafter_model_name)
        self.tokenizer.padding_side = "left"

    def _prepare_generation_config(self):
        assert self.model.generation_config is not None

        self.model.generation_config.return_dict_in_generate = True
        self.model.generation_config.output_scores = True
        self.model.generation_config.max_new_tokens = self.config.max_new_tokens

        match self.config.decode_method:
            case "greedy":
                self.model.generation_config.do_sample = False
            case "stochastic":
                self.model.generation_config.do_sample = True
                self.model.generation_config.num_beams = 1

    def draft(self, model_input: BatchEncoding) -> ModelOutputType:
        """Generate draft sequences from the model given tokenized input.

        Args:
            model_input (BatchEncoding): Tokenized input sequences.
        Returns:
            ModelOutputType: Generated draft sequences and associated data.
        """
        draft_config = transformers.GenerationConfig(
            max_new_tokens=self.config.draft_tokens_num,
        )
        draft: ModelOutputType = self.model.generate(
            **model_input, generation_config=draft_config, use_model_defaults=True  # type: ignore
        )
        return draft

    def tokenize(self, input_texts: list[str]) -> BatchEncoding:
        """Tokenize batch of input sequences using model's pre-trained tokenizer.

        Args:
            input_texts (list[str]): List of input strings to tokenize.

        Returns:
            BatchEncoding: Tokenized inputs as a BatchEncoding object.
        """
        return self.tokenizer(input_texts, return_tensors="pt", padding=True)

    def detokenize(self, token_ids: list[list[int]] | Tensor) -> list[str]:
        """Detokenize batch of token ID sequences back into strings.

        Args:
            token_ids (list[list[int]] | Tensor): Batch of token ID sequences.

        Returns:
            list[str]: Detokenized strings.
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(text, str)

        return text

    @property
    def device(self) -> torch.device:
        return self.model.device

    def __str__(self) -> str:
        string = f"Drafter(model_name={self.config.drafter_model_name})\nType: {type(self.model)}\n{self.model}"
        return string
