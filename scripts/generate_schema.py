import json
from pathlib import Path
from typing import TYPE_CHECKING

import docstring_parser as parser

from naive_speculate.config.external import UserSpeculateConfig

if TYPE_CHECKING:
    from pydantic import BaseModel


SCHEMA_PATH = Path(__file__).parent.parent / "config-schema.json"


def populate_field_description(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Populate field descriptions in a Pydantic model based on its docstring."""
    for submodel_info in model_cls.model_fields.values():
        # 1. Retrieve the submodel class, and parse its docstring
        submodel_cls = submodel_info.annotation

        if submodel_cls is None:
            continue

        submodel_doc = parser.parse(submodel_cls.__doc__ or "")
        name_to_description = {prop.arg_name: prop.description for prop in submodel_doc.params}

        # 2. Update field descriptions based on the parsed docstring
        for field_name, field_info in submodel_cls.model_fields.items():
            field_info.description = name_to_description.get(field_name, "")
            print("=================================")
            print(
                f"Updated {submodel_cls.__name__}.{field_name} description to:\n'{field_info.description}'"
            )
            print("=================================")
        submodel_cls.model_rebuild(force=True)

    model_cls.model_rebuild(force=True)

    return model_cls


def dump_schema(schema: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(schema, f, indent=2)


if __name__ == "__main__":
    print("Generating configuration schema...")
    model_cls = populate_field_description(UserSpeculateConfig)
    schema = model_cls.model_json_schema()
    dump_schema(schema, SCHEMA_PATH)
    print(f"Schema saved to {SCHEMA_PATH}")
