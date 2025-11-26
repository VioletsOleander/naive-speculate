import argparse
import sys

from torch import Tensor

from naive_speculate.modules.draft import Drafter
from naive_speculate.modules.verify import Verifier
from naive_speculate.utility import Config, Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Naive Speculate Inference")
    parser.add_argument("config_path", type=str, help="Path to the config file")

    return parser.parse_args()


def warmup(model: Drafter, sample_input: str, iterations: int = 3):
    model_input = model.tokenize([sample_input])

    for _ in range(iterations):
        _ = model.draft(model_input)


def speculate(
    drafter: Drafter, verifier: Verifier, input_text: str, verbose: bool = True
):
    # draft
    tokenized_input = drafter.tokenize([input_text])
    draft = drafter.draft(tokenized_input.to(drafter.device))

    if verbose:
        input_ids = tokenized_input["input_ids"]
        assert isinstance(input_ids, Tensor)
        context_length = input_ids.shape[1]
        draft_text = drafter.detokenize(draft.sequences[:, context_length:])[0]
        print(f"Drafted text:\n{draft_text}")

    # verify
    context_text = drafter.detokenize(draft.sequences)[0]
    tokenized_context = verifier.tokenize([context_text])

    verified_ids = verifier.verify(draft, tokenized_context.to(verifier.device))

    tokenized_input = verifier.tokenize([input_text])
    input_ids = tokenized_input["input_ids"]
    assert isinstance(input_ids, Tensor)
    context_length = input_ids.shape[1]
    verified_text = verifier.detokenize(verified_ids[:, context_length:])[0]

    print(f"Verified Text:\n{verified_text}")
    return verified_text


def main() -> int:
    try:
        args = parse_args()
        config = Config.from_file(args.config_path)

        drafter = Drafter(config)
        verifier = Verifier(config)

        prompt = """
        Summarize the key points of the following text:
        Once upon a time in a land far, far away, there lived a brave knight named Sir Lancelot.
        He was known throughout the kingdom for his courage, honor, and unwavering dedication to protecting the innocent.
        One day, Sir Lancelot received a quest from the king to rescue a princess who had been captured by a fearsome dragon.
        With his trusty sword and shield, Sir Lancelot set off on his journey, facing numerous challenges and dangers along the way.
        After a long and arduous battle, he finally defeated the dragon and rescued the princess, earning the gratitude of the entire kingdom.
        """
        # prompt = "Hello, who are you?"
        messages = [{"role": "user", "content": prompt}]
        input_text = drafter.apply_chat_template(messages)
        print(f"Input text:\n{input_text}")

        speculate(drafter, verifier, input_text)

        return 0
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
