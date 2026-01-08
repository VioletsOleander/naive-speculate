import json
from pathlib import Path


def load_context(context_path: str) -> list[dict[str, str]]:
    """Load context from specified file path."""
    with Path(context_path).open("r", encoding="utf-8") as f:
        context: list[dict[str, str]] = json.load(f)

    return context
