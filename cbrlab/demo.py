import json
from pathlib import Path
from cbrlab.io import load_cases, load_schema
from cbrlab.similarity.global import aggregate
from cbrlab.retrieval import Embedder, prepare_text

def main():
    ds = "data/device_faults"
    cases = load_cases(ds)
    texts = [c.get("text_problem") or prepare_text(c["problem"], c.get("text_problem")) for c in cases]
    emb = Embedder()
    emb.fit(texts)
    q = "Device error i59; output on; relay switched; Vdd 20.4"
    idxs, sims = emb.topk(q, k=1)
    top = cases[idxs[0]]
    schema = load_schema(ds)
    gscore, per = aggregate(top["problem"], top["problem"], schema)  # self-sim as example
    print("Query:", q)
    print("Top case:", top["id"], top["text_problem"])
    print("Vector sim:", sims[0], " Global sim (self):", round(gscore, 2))

if __name__ == "__main__":
    main()
