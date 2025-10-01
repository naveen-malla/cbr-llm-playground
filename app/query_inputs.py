from __future__ import annotations

from typing import Dict, Any, List

import streamlit as st


def _categorical_options(cases: List[dict], feature: str) -> List[str]:
    seen: List[str] = []
    for case in cases:
        value = case.get("problem", {}).get(feature)
        if value is None:
            continue
        sval = str(value)
        if sval not in seen:
            seen.append(sval)
    return seen


def collect_query_features(
    schema: Dict[str, Any],
    cases: List[dict],
    *,
    key_prefix: str = "query",
    show_header: bool = True,
) -> Dict[str, Any]:
    """Render Streamlit widgets for structured features and return the selections."""

    features = schema.get("features", {})
    default_problem = cases[0]["problem"] if cases else {}
    query: Dict[str, Any] = {}

    if show_header and features:
        st.markdown("#### Structured problem features")

    for name, meta in features.items():
        ftype = meta.get("type", "categorical")
        widget_key = f"{key_prefix}_{name}"
        default_value = default_problem.get(name)

        if ftype == "numeric":
            value = float(default_value) if default_value is not None else 0.0
            kwargs: Dict[str, Any] = {"value": value, "key": widget_key}
            if "range" in meta:
                min_v, max_v = meta["range"]
                kwargs["min_value"] = float(min_v)
                kwargs["max_value"] = float(max_v)
                span = float(max_v) - float(min_v)
                if span > 0:
                    step = meta.get("step")
                    if step is None:
                        step = span / 20.0 if span <= 20 else 1.0
                    kwargs["step"] = float(step)
            query[name] = float(
                st.number_input(
                    f"{name}",
                    help=meta.get("description"),
                    **kwargs,
                )
            )
        elif ftype == "boolean":
            checked = bool(default_value) if default_value is not None else False
            query[name] = bool(
                st.checkbox(
                    f"{name}",
                    value=checked,
                    help=meta.get("description"),
                    key=widget_key,
                )
            )
        else:
            options = _categorical_options(cases, name)
            default_str = str(default_value) if default_value is not None else (options[0] if options else "")
            if options:
                index = options.index(default_str) if default_str in options else 0
                query[name] = st.selectbox(
                    f"{name}",
                    options,
                    index=index,
                    help=meta.get("description"),
                    key=widget_key,
                )
            else:
                query[name] = st.text_input(
                    f"{name}",
                    value=default_str,
                    help=meta.get("description"),
                    key=widget_key,
                )

    return query
