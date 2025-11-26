get-models:
    uv run hf download Qwen/Qwen3-0.6B
    uv run hf download Qwen/Qwen3-1.7B

get-tools:
    uv tool install ruff
    uv tool install isort
    uv tool install ty

lint:
    uv run ruff check
    uv run ty check

format:
    uv run ruff format
    uv run isort .

format-check:
   uv run ruff format --check
   uv run isort --check .

check: format-check lint

infer config="config.example.toml":
    uv run infer {{config}}