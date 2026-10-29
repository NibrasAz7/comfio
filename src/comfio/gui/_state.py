"""Session-state keys and helpers for the comfio Streamlit app."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import streamlit as st

# Session-state keys
DF = "df"  # raw uploaded / demo / preloaded DataFrame
COLUMN_MAP = "column_map"  # canonical -> actual df column
TIMESTAMP_COL = "timestamp_col"  # optional timestamp column name
NAN_STRATEGY = "nan_strategy"  # interpolate | drop | fill_zero | raise
IP_UNITS = "ip_units"  # uploaded data declared in °F / fpm
CONFIG = "config"  # run_pipeline config dict
RESULT = "pipeline_result"  # PipelineResult after Act 2
COMPLIANCE = "compliance_report"  # ComplianceReport after Act 3
SCENARIOS = "scenarios"  # pinned run summaries for side-by-side compare


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any) -> None:  # noqa: A001
    st.session_state[key] = value


def get_df() -> pd.DataFrame | None:
    return get(DF)


def set_df(df: pd.DataFrame) -> None:
    """Store a new frame and invalidate downstream state."""
    set(DF, df)
    for k in (COLUMN_MAP, TIMESTAMP_COL, RESULT, COMPLIANCE):
        st.session_state.pop(k, None)


def get_column_map() -> dict[str, str]:
    return cast(dict[str, str], get(COLUMN_MAP, {}))


def set_column_map(mapping: dict[str, str]) -> None:
    set(COLUMN_MAP, dict(mapping))


def get_timestamp_col() -> str | None:
    return cast(str | None, get(TIMESTAMP_COL))


def is_mapped() -> bool:
    return bool(get(COLUMN_MAP))


def clear_mapping() -> None:
    for k in (COLUMN_MAP, TIMESTAMP_COL, RESULT, COMPLIANCE):
        st.session_state.pop(k, None)


def get_result() -> Any:
    return get(RESULT)


def has_result() -> bool:
    return get(RESULT) is not None


def reset() -> None:
    """Clear every GUI-managed key (back to the ingest screen)."""
    for k in (
        DF,
        COLUMN_MAP,
        TIMESTAMP_COL,
        NAN_STRATEGY,
        IP_UNITS,
        CONFIG,
        RESULT,
        COMPLIANCE,
    ):
        st.session_state.pop(k, None)


def get_scenarios() -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], get(SCENARIOS, []))


def pin_scenario(summary: dict[str, Any]) -> None:
    """Append a run summary to the compare list (max 4 kept)."""
    scenarios = get_scenarios()
    scenarios.append(summary)
    set(SCENARIOS, scenarios[-4:])


def clear_scenarios() -> None:
    st.session_state.pop(SCENARIOS, None)
