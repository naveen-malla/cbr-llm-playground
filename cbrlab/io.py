import json
from pathlib import Path
from typing import Any, Dict, List

def load_json(p: str | Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))

def load_cases(dirpath: str | Path) -> List[Dict]:
    dp = Path(dirpath)
    cases = load_json(dp / "cases.json")
    return cases

def load_schema(dirpath: str | Path) -> Dict:
    return load_json(Path(dirpath) / "schema.json")
