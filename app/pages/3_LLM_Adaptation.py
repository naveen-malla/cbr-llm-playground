import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.utils import render_feature_inputs
from cbrlab.io import load_cases, load_schema
from cbrlab.retrieval import Embedder, prepare_text
from cbrlab.adaptation.rules import load_rules, apply_rules
from cbrlab.adaptation.llm import propose_adaptation

ds = st.sidebar.selectbox("Dataset", ["device_faults","clinic_toy"])
base = Path("data") / ds
cases = load_cases(base)
schema = load_schema(base)
rules = load_rules(base / "rules.yaml")
texts = [c.get("text_problem") or prepare_text(c["problem"], c.get("text_problem")) for c in cases]
emb = Embedder()
emb.fit(texts)

st.subheader("LLM-assisted adaptation")
default_case = cases[0] if cases else {"problem": {}}
query_features = render_feature_inputs(
    schema,
    defaults=default_case.get("problem", {}),
    prefix="adapt",
)
st.caption("Structured inputs above feed the rule-based adaptation step.")

q = st.text_area("New problem:", value="Write your problem here...", height=100)
if st.button("Retrieve + Adapt"):
    idxs, _ = emb.topk(q, k=1)
    c = cases[idxs[0]]
    st.write("Retrieved case:", c["id"], c.get("text_problem"))
    st.json({"Query features": query_features})
    # Rule-based
    if isinstance(c["solution"], dict):
        base_solution = (
            c["solution"].get("steps")
            or c["solution"].get("advice")
            or str(c["solution"])
        )
    else:
        base_solution = str(c["solution"])
    rb = apply_rules(rules, query_features, c["problem"], base_solution)
    st.markdown("**Rule-based proposal:**")
    st.write(rb)
    # LLM-based
    try:
        la = propose_adaptation(q, c.get("text_problem", ""), rb)
        st.markdown("**LLM-adapted proposal (local, Ollama):**")
        st.write(la["adapted_solution"])
        st.caption(f"Model: {os.environ.get('OLLAMA_MODEL','phi3:mini')} (requires `ollama pull` and Ollama running)")
    except Exception as e:
        st.error(f"LLM call failed. Is Ollama running and model pulled? Error: {e}")
