get-models:
    uv run hf download Qwen/Qwen3-0.6B
    uv run hf download Qwen/Qwen3-1.7B

infer config="config.example.toml":
    uv run infer {{config}}