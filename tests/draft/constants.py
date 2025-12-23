PROMPT_LENGTH = 5
DRAFT_MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 15
DRAFT_TOKENS_NUM = 5
CONFIG_DICT = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "decode_method": "greedy",
    "drafter_model_name": DRAFT_MODEL_NAME,
    "draft_tokens_num": DRAFT_TOKENS_NUM,
    "verifier_model_name": DRAFT_MODEL_NAME,
}
