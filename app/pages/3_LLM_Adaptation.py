from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st

from app.data_loader import load_cases, load_schema
from app.rules import apply_rules

st.set_page_config(page_title="CBR LLM Adaptation", page_icon="🤖")

st.title("Rule-guided Adaptation Playground")
st.write(
    "Interactively adapt a retrieved case by describing the target query "
    "using the structured schema inputs."
)

schema = load_schema()
cases = load_cases()

case_lookup = {case["name"]: case for case in cases}
selected_case_name = st.selectbox("Retrieved case", options=list(case_lookup.keys()))
selected_case = case_lookup[selected_case_name]

st.subheader("Query description")
query_features: Dict[str, Any] = {}

for feature in schema.get("features", []):
    feature_name = feature["name"]
    display_name = feature.get("display_name", feature_name.replace("_", " ").title())
    feature_type = feature.get("type", "text")

    if feature_type == "number":
        number_kwargs: Dict[str, Any] = {"value": feature.get("default", 0)}

        for bound in ("min_value", "max_value", "step"):
            if feature.get(bound) is not None:
                number_kwargs[bound] = feature[bound]

        is_float_input = any(
            isinstance(number_kwargs.get(key), float) for key in ("value", "min_value", "max_value", "step")
        )

        if is_float_input:
            number_kwargs.setdefault("format", "%0.2f")
            for key in ("value", "min_value", "max_value", "step"):
                if key in number_kwargs and number_kwargs[key] is not None:
                    number_kwargs[key] = float(number_kwargs[key])
        else:
            number_kwargs.setdefault("format", "%d")
            for key in ("value", "min_value", "max_value", "step"):
                if key in number_kwargs and number_kwargs[key] is not None:
                    number_kwargs[key] = int(number_kwargs[key])

        query_features[feature_name] = st.number_input(display_name, **number_kwargs)
    elif feature_type == "select":
        options = feature.get("options", [])

        if not options:
            query_features[feature_name] = st.text_input(display_name, value=feature.get("default", ""))
        else:
            default = feature.get("default", options[0])
            default_index = options.index(default) if default in options else 0
            query_features[feature_name] = st.selectbox(
                display_name, options=options, index=default_index
            )
    else:
        query_features[feature_name] = st.text_input(display_name, value=feature.get("default", ""))

st.caption(
    "The captured query features will be supplied to the rule engine so you can "
    "see how each attribute influences the adaptation."
)

adapted_case, adaptation_notes = apply_rules(selected_case, query_features)

st.subheader("Query features snapshot")
st.json(query_features)

st.subheader("Adapted case")
st.json(adapted_case)

st.subheader("Adaptation notes")
if adaptation_notes:
    for note in adaptation_notes:
        st.markdown(f"- {note}")
else:
    st.info("No rule adjustments were necessary for the selected query.")
