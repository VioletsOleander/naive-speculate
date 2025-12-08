import pytest
from torch import Tensor

from naive_speculate.utils import SpeculateConfig, Tokenizer

CONFIG_DICT = {
    "max_new_tokens": 32768,
    "decode_method": "greedy",
    "drafter_model_name": "Qwen/Qwen3-0.6B",
    "draft_tokens_num": 32768,
    "verifier_model_name": "Qwen/Qwen3-1.7B",
}

MESSAGES = [
    {"role": "user", "content": "Hello, who are you?"},
    {
        "role": "assistant",
        "content": "I am an AI language model developed to assist you.",
    },
    {"role": "user", "content": "What can you do for me?"},
    {
        "role": "assistant",
        "content": "I can help you with a variety of tasks, such as answering questions, providing explanations, and generating text.",
    },
]


@pytest.fixture
def tokenizer() -> Tokenizer:
    config = SpeculateConfig.from_dict(CONFIG_DICT)
    return Tokenizer(config)


@pytest.fixture
def text(tokenizer: Tokenizer) -> str:
    return tokenizer.apply_chat_template(MESSAGES)


def test_tokenizer(tokenizer: Tokenizer, text: str) -> None:
    tokenized = tokenizer.tokenize([text])

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    assert isinstance(input_ids, Tensor)
    assert isinstance(attention_mask, Tensor)

    detokenized = tokenizer.detokenize(input_ids.tolist())[0]

    assert detokenized == text
