import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.utils import render_feature_inputs
from cbrlab.io import load_cases, load_schema
from cbrlab.retrieval import Embedder, prepare_text
from cbrlab.similarity.global_similarity import aggregate
from cbrlab.explain.tables import per_feature_table

ds = st.sidebar.selectbox("Dataset", ["device_faults","clinic_toy"])
base = Path("data") / ds
cases = load_cases(base)
schema = load_schema(base)
texts = [c.get("text_problem") or prepare_text(c["problem"], c.get("text_problem")) for c in cases]
emb = Embedder()
emb.fit(texts)

st.subheader("Retrieve similar cases")
default_case = cases[0] if cases else {"problem": {}}
query_features = render_feature_inputs(
    schema,
    defaults=default_case.get("problem", {}),
    prefix="retrieve",
)
st.caption("Set the structured query features above; they drive the symbolic comparison.")

q_default = texts[0] if texts else ""
q = st.text_area("Describe your problem:", value=q_default, height=100)
k = st.slider("Top-k", 1, 5, 3)

idxs, sims = emb.topk(q, k=k)
schema = load_schema(base)

for rank, (i, s) in enumerate(zip(idxs, sims), start=1):
    c = cases[i]
    gscore, per = aggregate(query_features, c["problem"], schema)
    st.markdown(f"### #{rank} — {c['id']}")
    st.write("Case text:", c.get("text_problem"))
    st.write("Vector similarity:", round(float(s), 3))
    st.write("Global similarity:", round(float(gscore), 3))
    st.dataframe(per_feature_table(per))
