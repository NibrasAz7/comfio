"""Sidebar configuration widgets for the comfio Streamlit app.

Renders the parameter panel and returns a ``run_pipeline``-compatible
config dict. ``met``/``clo`` are parameters, not data columns — matching
how the personas' real files look (they are never measured).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from comfio.gui import _state


def render_sidebar(df_columns: set[str]) -> dict[str, Any]:
    """Render the configuration sidebar. Returns the pipeline config."""
    st.sidebar.header("Parameters")

    # Assumed-occupant parameters — these are never columns in real data.
    with st.sidebar.expander("Occupant assumptions", expanded=False):
        met = st.number_input(
            "Metabolic rate [met]",
            min_value=0.8,
            max_value=4.0,
            value=1.2,
            step=0.1,
            help="Activity level: 1.0 seated · 1.2 office work · 2.0 walking",
        )
        clo = st.number_input(
            "Clothing insulation [clo]",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="0.5 summer · 1.0 winter business attire",
        )
        vr = st.slider(
            "Air speed [m/s] (if not measured)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Used only when no air-velocity column is mapped.",
        )
        mrt_equals_tdb = st.checkbox(
            "Assume radiant temp = air temp",
            value=True,
            help="Standard assumption when no MRT channel exists.",
        )

    st.sidebar.header("Standards")
    category = st.sidebar.selectbox(
        "Thermal category (ISO 7730)", ["B", "A", "C"], index=0
    )
    task_type = st.sidebar.selectbox(
        "Visual task (EN 12464-1)",
        ["general", "office", "reading", "corridor", "precision"],
        index=0,
    )
    nc_level = st.sidebar.selectbox(
        "Acoustic level", ["NC-25", "NC-35", "NC-40", "NC-50"], index=1
    )
    co2_threshold = st.sidebar.selectbox(
        "IAQ CO₂ level", ["excellent", "good", "acceptable"], index=1
    )
    weight_preset = st.sidebar.selectbox(
        "IEQ weighting preset",
        ["default", "equal", "office", "school", "healthcare"],
        index=0,
    )

    nan_strategy = st.sidebar.selectbox(
        "NaN handling",
        ["interpolate", "drop", "fill_zero"],
        index=0,
        help="How to treat missing values in the sensor columns.",
    )
    _state.set(_state.NAN_STRATEGY, nan_strategy)

    cfg: dict[str, Any] = {
        "met": met,
        "clo": clo,
        "vr": vr,
        "category": category,
        "task_type": task_type,
        "nc_level": nc_level,
        "co2_threshold": co2_threshold,
        "weight_preset": weight_preset,
        "threshold": 80.0,
    }
    # Extras consumed by the GUI layer, not run_pipeline.
    cfg["_mrt_equals_tdb"] = mrt_equals_tdb
    cfg["_has_outdoor"] = "outdoor_temp_c" in df_columns
    return cfg


def render_evidence_params() -> tuple[float, str]:
    """Act-3 compliance parameters. Returns (threshold, zone_id)."""
    c1, c2 = st.columns(2)
    threshold = c1.number_input(
        "Compliance threshold (IEQ Index)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=5.0,
    )
    zone_id = c2.text_input("Zone / space ID", value="", placeholder="e.g. Zone-3 North")
    return float(threshold), zone_id.strip() or "zone-1"
