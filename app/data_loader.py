from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_schema() -> Dict[str, Any]:
    """Return the dataset schema used to render query inputs."""

    schema_path = _DATA_DIR / "schema.json"
    return _read_json(schema_path)


def load_cases() -> List[Dict[str, Any]]:
    """Return the library of source cases for the demo application."""

    cases_path = _DATA_DIR / "cases.json"
    return _read_json(cases_path)


__all__ = ["load_schema", "load_cases"]
