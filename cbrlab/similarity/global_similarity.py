from typing import Dict
from .local import sim_numeric, sim_bool, sim_cat

def aggregate(problem_q: dict, problem_c: dict, schema: dict) -> tuple[float, Dict[str, float]]:
    feats = schema["features"]
    weights = schema.get("global_weights", {})
    per_feature = {}
    total = 0.0
    wsum = 0.0
    for k, meta in feats.items():
        w = float(weights.get(k, 1.0))
        a = problem_q.get(k, None)
        b = problem_c.get(k, None)
        if a is None or b is None:
            s = 0.0
        else:
            t = meta["type"]
            if t == "numeric":
                s = sim_numeric(float(a), float(b), tuple(meta.get("range", (0.0, 1.0))))
            elif t == "boolean":
                s = sim_bool(bool(a), bool(b))
            else:
                s = sim_cat(str(a), str(b))
        per_feature[k] = s
        total += w * s
        wsum += w
    return (total / wsum if wsum > 0 else 0.0, per_feature)
