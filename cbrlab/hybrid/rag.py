try:  # optional dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None


def _normalize(values):
    if np is not None:
        arr = np.asarray(values, dtype=float)
        span = arr.max() - arr.min()
        if span == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / (span + 1e-9)
    data = [float(v) for v in values]
    lo = min(data)
    hi = max(data)
    span = hi - lo
    if span == 0:
        return [0.0 for _ in data]
    return [(v - lo) / (span + 1e-9) for v in data]


def fuse(symbolic_scores, vector_scores, alpha: float = 0.5):
    # assume both are lists aligned by candidate order
    sym = _normalize(symbolic_scores)
    vec = _normalize(vector_scores)
    if np is not None and isinstance(sym, np.ndarray):
        fused = alpha * sym + (1 - alpha) * vec
        order = np.argsort(fused)[::-1]
        return order.tolist(), fused[order].tolist()

    combined = [alpha * s + (1 - alpha) * v for s, v in zip(sym, vec)]
    order = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)
    return order, [combined[i] for i in order]
