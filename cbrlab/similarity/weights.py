def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    s = sum(weights.values()) or 1.0
    return {k: v/s for k, v in weights.items()}
