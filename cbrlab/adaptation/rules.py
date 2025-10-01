from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for offline mode
    yaml = None

def load_rules(path: str | Path):
    p = Path(path)
    if not p.exists() or yaml is None:
        return []
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def apply_rules(rules, query: dict, case: dict, solution_text: str) -> str:
    augmented = solution_text
    for r in rules:
        if r.get("action") == "append_note" and r.get("if"):
            # very simple "if": only supports voltage example; safe eval avoided for pedagogy
            try:
                qv = query.get("voltage_vdd")
                cv = case.get("voltage_vdd")
                cond = abs(float(qv) - float(cv)) > 3.0
            except Exception:
                cond = False
            if cond:
                augmented += f" Note: {r['params'].get('note','')}"
    return augmented
