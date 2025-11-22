get-models:
    uv run hf download google-t5/t5-small
    uv run hf download google-t5/t5-base
    uv run hf download google-t5/t5-large

infer config="config.toml":
    uv run infer {{config}}