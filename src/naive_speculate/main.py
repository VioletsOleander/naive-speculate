import argparse

from .dependency import DependencyContainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive Speculative Decoding")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("context_file", type=str, help="Path to the context file.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dependencies = DependencyContainer(config_path=args.config_file, context_path=args.context_file)

    tokenizer = dependencies.tokenizer
    context = dependencies.context

    prompt = tokenizer.apply_chat_template(context)
    input_ids, _attention_mask = tokenizer.tokenize([prompt])

    speculative_decoder = dependencies.speculative_decoder
    _output_ids = speculative_decoder.speculative_decode(
        query_token_ids=input_ids,
        draft_strategy=dependencies.draft_strategy,
        verify_strategy=dependencies.verify_strategy,
    )

    return 0
