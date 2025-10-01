from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

Case = Dict[str, Any]
FeatureMap = Dict[str, Any]


def apply_rules(case: Case, query_features: FeatureMap) -> Tuple[Case, List[str]]:
    """Adapt *case* to the *query_features* using rule-based tweaks.

    Parameters
    ----------
    case:
        The retrieved case that will act as the baseline adaptation candidate.
    query_features:
        Structured features captured from the Streamlit form.

    Returns
    -------
    (Case, List[str])
        A tuple containing the adapted case and notes describing the
        transformations applied.
    """

    adapted_case: Case = deepcopy(case)
    notes: List[str] = []

    _apply_voltage_rule(adapted_case, query_features, notes)
    _apply_environment_rule(adapted_case, query_features, notes)

    return adapted_case, notes


def _apply_voltage_rule(adapted_case: Case, query_features: FeatureMap, notes: List[str]) -> None:
    case_voltage = adapted_case.get("voltage_rating")
    query_voltage = query_features.get("voltage_rating")

    if query_voltage is None:
        return

    if case_voltage != query_voltage:
        notes.append(
            "Adjusted voltage rating from "
            f"{case_voltage if case_voltage is not None else 'unknown'}V to {query_voltage}V to satisfy the query."
        )
        adapted_case["voltage_rating"] = query_voltage


def _apply_environment_rule(adapted_case: Case, query_features: FeatureMap, notes: List[str]) -> None:
    case_environment = adapted_case.get("environment")
    query_environment = query_features.get("environment")

    if query_environment is None or case_environment == query_environment:
        return

    notes.append(
        "Updated environmental rating from "
        f"{case_environment or 'unspecified'} to {query_environment} for deployment compatibility."
    )
    adapted_case["environment"] = query_environment
