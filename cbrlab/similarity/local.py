def sim_numeric(a: float, b: float, rng: tuple[float, float]) -> float:
    lo, hi = rng
    if hi <= lo:
        return 1.0 if a == b else 0.0
    return 1.0 - min(abs(a - b) / (hi - lo), 1.0)

def sim_bool(a: bool, b: bool) -> float:
    return 1.0 if bool(a) == bool(b) else 0.0

def sim_cat(a: str, b: str) -> float:
    return 1.0 if str(a) == str(b) else 0.0
