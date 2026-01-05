get-tools:
    uv tool install ty
    uv tool install pylint

gen-schema:
    uv run python scripts/generate_schema.py

format:
    uv run ruff format src tests scripts

lint:
    uv run ruff check src tests scripts

type-check:
    uv run ty check src tests scripts
