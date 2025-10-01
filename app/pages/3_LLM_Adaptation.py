import os, json
import streamlit as st
from pathlib import Path
from cbrlab.io import load_cases, load_schema
from cbrlab.retrieval import Embedder, prepare_text
from cbrlab.adaptation.rules import load_rules, apply_rules
from cbrlab.adaptation.llm import propose_adaptation

ds = st.sidebar.selectbox("Dataset", ["device_faults","clinic_toy"])
base = Path("data") / ds
cases = load_cases(base)
rules = load_rules(base / "rules.yaml")
texts = [c.get("text_problem") or prepare_text(c["problem"], c.get("text_problem")) for c in cases]
emb = Embedder(); emb.fit(texts)

st.subheader("LLM-assisted adaptation")
q = st.text_area("New problem:", value="Write your problem here...", height=100)
if st.button("Retrieve + Adapt"):
    idxs, _ = emb.topk(q, k=1)
    c = cases[idxs[0]]
    st.write("Retrieved case:", c["id"], c.get("text_problem"))
    # Rule-based
    rb = apply_rules(rules, {}, c["problem"], c["solution"]["steps"] if "steps" in c["solution"] else str(c["solution"]))
    st.markdown("**Rule-based proposal:**")
    st.write(rb)
    # LLM-based
    try:
        la = propose_adaptation(q, c.get("text_problem",""), rb)
        st.markdown("**LLM-adapted proposal (local, Ollama):**")
        st.write(la["adapted_solution"])
        st.caption(f"Model: {os.environ.get('OLLAMA_MODEL','phi3:mini')} (requires `ollama pull` and Ollama running)")
    except Exception as e:
        st.error(f"LLM call failed. Is Ollama running and model pulled? Error: {e}")
