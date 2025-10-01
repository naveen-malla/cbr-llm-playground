import pandas as pd
import streamlit as st
from pathlib import Path

from app.query_inputs import collect_query_features
from cbrlab.hybrid.rag import fuse
from cbrlab.io import load_cases, load_schema
from cbrlab.retrieval import Embedder, prepare_text
from cbrlab.similarity import aggregate


ds = st.sidebar.selectbox("Dataset", ["device_faults", "clinic_toy"])
base = Path("data") / ds
cases = load_cases(base)
schema = load_schema(base)
texts = [c.get("text_problem") or prepare_text(c["problem"], c.get("text_problem")) for c in cases]
emb = Embedder()
emb.fit(texts)

st.subheader("Hybrid symbolic + vector retrieval")
q = st.text_area("Query", value=texts[0], height=100)
query_features = collect_query_features(schema, cases, key_prefix="hybrid")
alpha = st.slider("Fusion weight α (symbolic)", 0.0, 1.0, 0.5, 0.05)
k = st.slider("Top-k", 1, 5, 3)

idxs, sims = emb.topk(q, k=len(cases))

sym_scores = [aggregate(query_features, c["problem"], schema)[0] for c in cases]
vec_scores = [float(v) for v in sims]
sym_sel = [sym_scores[i] for i in idxs]
vec_sel = [vec_scores[i] for i in idxs]
order, fused = fuse(sym_sel, vec_sel, alpha=alpha)

st.caption(f"Fused score = α·symbolic + (1-α)·vector after normalization (α = {alpha:.2f}).")
st.write("Showing top-k after fusion:")

top_results = []
for r in range(min(k, len(order))):
    case_idx = idxs[order[r]]
    top_results.append(
        {
            "rank": r + 1,
            "case_index": case_idx,
            "case_id": cases[case_idx]["id"],
            "symbolic": sym_sel[order[r]],
            "vector": vec_sel[order[r]],
            "fused": fused[r],
        }
    )

if top_results:
    display_df = pd.DataFrame(top_results).set_index("rank")
    display_df = display_df.drop(columns=["case_index"])
    display_df[["symbolic", "vector", "fused"]] = display_df[["symbolic", "vector", "fused"]].round(3)
    st.dataframe(display_df, use_container_width=True)

    for row in top_results:
        case = cases[row["case_index"]]
        st.markdown(
            f"**#{row['rank']}** {case['id']} — fused={row['fused']:.3f}, sym={row['symbolic']:.3f}, vec={row['vector']:.3f}"
        )
        st.caption(case.get("text_problem"))
else:
    st.info("Adjust the query or dataset to generate comparable cases.")
