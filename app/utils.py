"""Utility helpers shared across Streamlit pages."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import streamlit as st


def _numeric_bounds(meta: Mapping[str, Any]) -> tuple[float, float]:
    rng = meta.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        try:
            lo = float(rng[0])
            hi = float(rng[1])
            if hi <= lo:
                hi = lo + 1.0
            return lo, hi
        except (TypeError, ValueError):
            pass
    return 0.0, 1.0


def render_feature_inputs(
    schema: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
    *,
    prefix: str = "query",
) -> Dict[str, Any]:
    """Render Streamlit widgets for each schema feature and return the collected values."""

    features: Dict[str, Any] = {}
    defaults = defaults or {}

    for name, meta in schema.get("features", {}).items():
        key = f"{prefix}_{name}"
        ftype = meta.get("type", "categorical")
        default_val = defaults.get(name)

        if ftype == "numeric":
            lo, hi = _numeric_bounds(meta)
            if default_val is None:
                default_val = (lo + hi) / 2.0
            try:
                value = float(default_val)
            except (TypeError, ValueError):
                value = (lo + hi) / 2.0
            step = 0.1 if hi - lo <= 10 else 1.0
            features[name] = st.number_input(
                f"{name} ({ftype})",
                value=float(value),
                min_value=float(lo),
                max_value=float(hi),
                step=float(step),
                key=key,
            )
        elif ftype == "boolean":
            features[name] = st.checkbox(
                f"{name} ({ftype})",
                value=bool(default_val) if default_val is not None else False,
                key=key,
            )
        else:
            text_default = "" if default_val is None else str(default_val)
            features[name] = st.text_input(
                f"{name} ({ftype})",
                value=text_default,
                key=key,
            )

    return features

