def rationale(sim_score: float, k: int) -> str:
    return f"Retrieved top-{k} case with global similarity {sim_score:.2f}. Differences explained in per-feature table."
