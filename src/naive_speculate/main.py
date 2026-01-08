import argparse
import logging

from .config.load import load_config
from .dependency import DependencyContainer
from .utils.context import load_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive Speculative Decoding")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("context_file", type=str, help="Path to the context file.")

    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        speculate_config = load_config(config_path=args.config_file)
        dependencies = DependencyContainer(config=speculate_config)

        tokenizer = dependencies.tokenizer
        context = load_context(context_path=args.context_file)
        prompt = tokenizer.apply_chat_template(messages=context)
        input_ids, _attention_mask = tokenizer.tokenize(input_texts=[prompt])

        speculative_decoder = dependencies.speculative_decoder
        _output_ids = speculative_decoder.speculative_decode(
            query_token_ids=input_ids,
            num_draft_tokens=speculate_config.draft.num_draft_tokens,
            sample_strategy=speculate_config.draft.sample_strategy,
            verify_strategy=speculate_config.verify.verify_strategy,
        )

    except Exception:
        logging.exception("An error occurred during speculative decoding.")  # noqa: LOG015
        return 1
    else:
        return 0
