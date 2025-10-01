import json
import streamlit as st
from pathlib import Path
from cbrlab.predicates import av_to_predicates
import networkx as nx

def load_ds(name: str):
    base = Path("data") / name
    cases = json.loads((base / "cases.json").read_text(encoding="utf-8"))
    schema = json.loads((base / "schema.json").read_text(encoding="utf-8"))
    return cases, schema

ds = st.sidebar.selectbox("Dataset", ["device_faults","clinic_toy"])
cases, schema = load_ds(ds)
st.subheader(f"Cases: {ds}")
for c in cases:
    with st.expander(c["id"]):
        st.write("Problem:", c["problem"])
        st.write("Text problem:", c.get("text_problem",""))
        st.write("Solution:", c["solution"])
        st.caption("Predicates: " + ", ".join(av_to_predicates(c["problem"], c["id"])))

# toy graph view
G = nx.Graph()
for c in cases:
    for k in c["problem"].keys():
        G.add_edge(c["id"], k)
st.subheader("Tiny graph projection (case ↔ features)")
st.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
