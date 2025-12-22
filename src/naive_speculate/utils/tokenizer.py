from typing import Literal, overload

from torch import Tensor
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast

from .config import SpeculateConfig


class Tokenizer:
    """Wrapper class for tokenizer of drafter and verifier.

    It is assumed that drafter and verifier share the same tokenizer.

    Attributes:
        tokenizer (PreTrainedTokenizerFast): The tokenizer instance.
    """

    tokenizer: PreTrainedTokenizerFast

    def __init__(self, config: SpeculateConfig) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.drafter_model_name, local_files_only=True
        )
        assert isinstance(self.tokenizer, PreTrainedTokenizerFast)
        self.tokenizer.padding_side = "left"

    @overload
    def tokenize(
        self, input_texts: list[str], return_tensors: Literal[True] = True
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def tokenize(
        self, input_texts: list[str], return_tensors: Literal[False]
    ) -> BatchEncoding: ...

    @overload
    def tokenize(
        self,
        input_texts: list[str],
        return_tensors: bool,
    ) -> BatchEncoding | tuple[Tensor, Tensor]: ...

    def tokenize(
        self, input_texts: list[str], return_tensors: bool = True
    ) -> BatchEncoding | tuple[Tensor, Tensor]:
        """Tokenize a batch of input sequences into token ID sequences.

        Returns either a BatchEncoding object or a tuple of tensors (input_ids and attention_mask) based on return_tensors flag.

        Args:
            input_texts (list[str]): List of input strings to tokenize.
            return_tensors (bool): Whether to return tensors or BatchEncoding object.

        Returns:
            BatchEncoding | tuple[Tensor, Tensor]: Tokenized output.
        """
        tokenized = self.tokenizer(input_texts, return_tensors="pt", padding=True)

        if not return_tensors:
            return tokenized

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        assert isinstance(input_ids, Tensor)
        assert isinstance(attention_mask, Tensor)
        return input_ids, attention_mask

    def detokenize(
        self, token_ids: Tensor, skip_special_tokens: bool = False
    ) -> list[str]:
        """Detokenize a batch of token ID sequences back into strings.

        Args:
            token_ids (Tensor): Batch of token ID sequences. Shape [batch_size, seq_len].
            skip_special_tokens (bool): Whether to skip special tokens during decoding. Defaults to False.

        Returns:
            list[str]: Detokenized strings. Length [batch_size].
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
            enable_thinking (bool): Whether to enable thinking mode in the chat template. Defaults to True.

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
