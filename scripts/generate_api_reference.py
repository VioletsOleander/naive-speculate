from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR = DOCS_DIR / "reference"


def generate_api_reference(source_dir: Path, output_dir: Path) -> None:
    """Generate API reference Markdown files for mkdocstrings, based on the source code structure.

    Args:
        source_dir (Path): The root directory of the source code.
        output_dir (Path): The directory where the generated Markdown files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for module_path in source_dir.rglob("*.py"):
        if module_path.name.startswith("_") and module_path.name != "__init__.py":
            continue

        # 1. Determine the output path and fully qualified name for each module
        relative_path = module_path.relative_to(source_dir)
        module_parts = relative_path.with_suffix("").parts

        # __init__.py to index.md, <other>.py to <other>.md
        # name of __init__.py is the package name
        if module_parts[-1] == "__init__":
            output_path = output_dir.joinpath(*module_parts[:-1], "index.md")
            module_full_name = ".".join(module_parts[:-1])
        else:
            output_path = output_dir.joinpath(*module_parts).with_suffix(".md")
            module_full_name = ".".join(module_parts)

        # 2. Write the mkdocstrings directive to the output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"# {module_full_name}\n\n")
            f.write(f"::: {module_full_name}\n")

        print(f"Generated: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    print("Generating API reference...")
    generate_api_reference(SOURCE_DIR, OUTPUT_DIR)
    print("API reference generation complete.")
