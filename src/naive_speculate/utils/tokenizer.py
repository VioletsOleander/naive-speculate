from typing import TypeAlias

from transformers import BatchEncoding, Qwen2TokenizerFast

from .config import SpeculateConfig

TokenizerType: TypeAlias = Qwen2TokenizerFast


class Tokenizer:
    """
    Wrapper class for tokenizer of drafter and verifier. It is assumed that drafter and verifier
    shared the same tokenizer.
    """

    tokenizer: TokenizerType

    def __init__(self, config: SpeculateConfig):
        self.tokenizer = TokenizerType.from_pretrained(
            config.drafter_model_name, local_files_only=True
        )
        self.tokenizer.padding_side = "left"

    def tokenize(self, input_texts: list[str]) -> BatchEncoding:
        """Tokenize a batch of input sequences into token ID sequences.

        Args:
            input_texts (list[str]): List of input strings to tokenize.

        Returns:
            BatchEncoding: Tokenized inputs as a BatchEncoding object.
        """
        return self.tokenizer(input_texts, return_tensors="pt", padding=True)

    def detokenize(
        self, token_ids: list[list[int]], skip_special_tokens: bool = False
    ) -> list[str]:
        """Detokenize a batch of token ID sequences back into strings.

        Args:
            token_ids (list[list[int]] | Tensor): Batch of token ID sequences.

        Returns:
            list[str]: Detokenized strings.
        """
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def apply_chat_template(
        self, messages: list[dict[str, str]], enable_thinking: bool = True
    ) -> str:
        """Construct prompt text from chat messages using the tokenizer's chat template.

        Args:
            messages (list[dict[str, str]]): List of chat messages, where each message is a dict
                with keys "role" and "content".

        Returns:
            str: Constructed prompt text.
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        assert isinstance(text, str)

        return text
