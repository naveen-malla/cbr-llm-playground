import json, numpy as np
import streamlit as st
from pathlib import Path
from cbrlab.io import load_cases, load_schema
from cbrlab.retrieval import Embedder, prepare_text
from cbrlab.similarity.global import aggregate
from cbrlab.explain.tables import per_feature_table

ds = st.sidebar.selectbox("Dataset", ["device_faults","clinic_toy"])
base = Path("data") / ds
cases = load_cases(base)
texts = [c.get("text_problem") or prepare_text(c["problem"], c.get("text_problem")) for c in cases]
emb = Embedder()
emb.fit(texts)

st.subheader("Retrieve similar cases")
q = st.text_area("Describe your problem:", value=texts[0], height=100)
k = st.slider("Top-k", 1, 5, 3)

idxs, sims = emb.topk(q, k=k)
schema = load_schema(base)

for rank, (i, s) in enumerate(zip(idxs, sims), start=1):
    c = cases[i]
    gscore, per = aggregate(c["problem"], c["problem"], schema)  # self-sim to show table
    st.markdown(f"### #{rank} — {c['id']}")
    st.write("Case text:", c.get("text_problem"))
    st.write("Vector similarity:", round(float(s), 3))
    st.dataframe(per_feature_table(per))
