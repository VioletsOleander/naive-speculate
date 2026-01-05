get-tools:
    uv tool install ty
    uv tool install pylint

gen-schema:
    uv run --group dev scripts/generate_schema.py

gen-api-ref:
    uv run --group dev scripts/generate_api_reference.py

format:
    uv run ruff format src tests scripts

lint:
    uv run ruff check src tests scripts

type-check:
    uv run ty check src tests scripts
