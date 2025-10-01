def av_to_predicates(av: dict, case_id: str | None = None) -> list[str]:
    # attribute-value -> predicate literals: attr(value) or attr(key,value)
    preds = []
    for k, v in av.items():
        lit = f"{k}({str(v).replace(' ','_')})"
        preds.append(lit if case_id is None else f"{lit}@{case_id}")
    return preds
