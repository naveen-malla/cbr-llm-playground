import json, numpy as np, streamlit as st
from pathlib import Path
from cbrlab.io import load_cases, load_schema
from cbrlab.retrieval import Embedder, prepare_text
from cbrlab.similarity.global import aggregate
from cbrlab.hybrid.rag import fuse

ds = st.sidebar.selectbox("Dataset", ["device_faults","clinic_toy"])
base = Path("data") / ds
cases = load_cases(base)
schema = load_schema(base)
texts = [c.get("text_problem") or prepare_text(c["problem"], c.get("text_problem")) for c in cases]
emb = Embedder()
emb.fit(texts)

st.subheader("Hybrid symbolic + vector retrieval")
q = st.text_area("Query", value=texts[0], height=100)
alpha = st.slider("Fusion weight α (symbolic)", 0.0, 1.0, 0.5, 0.05)
k = st.slider("Top-k", 1, 5, 3)

idxs, sims = emb.topk(q, k=len(cases))
# compute symbolic scores vs query using global similarity (problem-only demo)
def symbolic_score(case):
    s, _ = aggregate(case["problem"], case["problem"], schema)  # self-sim placeholder
    return s
sym_scores = [symbolic_score(c) for c in cases]
sym_sel = [sym_scores[i] for i in idxs]
vec_sel = sims
order, fused = fuse(sym_sel, vec_sel, alpha=alpha)
st.write("Showing top-k after fusion:")
for r in range(min(k, len(order))):
    i = idxs[order[r]]
    st.markdown(f"**#{r+1}** {cases[i]['id']} — fused={fused[r]:.3f}, sym={sym_sel[order[r]]:.3f}, vec={vec_sel[order[r]]:.3f}")
    st.caption(cases[i].get("text_problem"))
