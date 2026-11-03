"""Global database benchmark — ASHRAE DB II (CC0) vs your building.

The ~3 MB dataset is fetched only after an explicit button click, cached
sha256-verified under ``$COMFIO_CACHE_DIR`` / ``~/.cache/comfio``, and
served from cache on every later rerun.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from comfio.gui import _charts, _state
from comfio.gui.tools.benchmark import _logic
from comfio.integration import comfort_db
from comfio.integration.comfort_db import DB2_CITATION, ComfortDBUnavailable

_CC0_NOTE = "data CC0-1.0 (doi:10.6078/D1F671)"


@st.cache_data(show_spinner="Downloading ASHRAE DB II (≈3 MB)…")
def _load_db() -> pd.DataFrame:
    return comfort_db.load_ashrae_db2()


def _filters(db: pd.DataFrame) -> pd.DataFrame:
    """One multiselect per FILTERS column; returns the filtered frame."""
    cols = st.columns(len(_logic.FILTERS))
    selected: dict[str, list[str]] = {}
    for i, col in enumerate(_logic.FILTERS):
        if col not in db.columns:
            continue
        options = sorted(db[col].dropna().unique().tolist())
        selected[col] = cols[i].multiselect(
            col.replace("_", " "), options, key=f"bench_f_{col}"
        )
    return _logic.filter_db(db, **selected)


def _user_overlay() -> pd.DataFrame | None:
    """Workbench data mapped onto the DB scatter columns, if available."""
    df = _state.get_df()
    mapping = _state.get_column_map()
    if df is None or not mapping:
        return None
    return _logic.user_overlay(df, mapping)


def render() -> None:
    st.title("Global database benchmark")
    st.caption(
        "Benchmark your building against the ASHRAE Global Thermal "
        "Comfort Database II (~109k votes, 809 buildings)."
    )
    st.caption(f"{DB2_CITATION} · {_CC0_NOTE}")

    requested = bool(st.session_state.get("bench_db_requested"))
    if st.button("Fetch database (≈3 MB — needs internet)", type="primary"):
        st.session_state["bench_db_requested"] = True
        requested = True
    if not requested:
        st.info(
            "Click 'Fetch database' to download the dataset once into the "
            "local cache — nothing is fetched without it."
        )
        return
    try:
        db = _load_db()
    except ComfortDBUnavailable as exc:
        st.error(str(exc))
        return

    db = _logic.with_operative(db)
    n_buildings = (
        db["building_id"].nunique() if "building_id" in db.columns else "—"
    )
    st.caption(f"{len(db):,} measurements · {n_buildings} buildings")
    filtered = _filters(db)
    st.caption(f"{len(filtered):,} rows after filters")

    tab_scatter, tab_bins, tab_bias = st.tabs(
        ["Adaptive scatter", "Sensation vs operative", "PMV bias"]
    )

    with tab_scatter:
        overlay = _user_overlay()
        st.plotly_chart(
            _charts.db_adaptive_scatter(filtered, overlay), width="stretch"
        )
        st.caption(
            "The outdoor axis is the database's 7-day running mean "
            "(t_mot_isd) — not the ASHRAE prevailing mean — so the ASHRAE "
            "adaptive band is drawn for reference only. Large selections "
            "are subsampled to 20k points for rendering."
        )
        if overlay is None:
            st.caption(
                "Load and map your data in 'Analyse my data' to plot it "
                "here as black stars."
            )

    with tab_bins:
        bins = _logic.sensation_bins(filtered)
        if bins.empty:
            st.info("No operative temperature data for these filters.")
        else:
            st.plotly_chart(
                _charts.sensation_bins_chart(bins), width="stretch"
            )

    with tab_bias:
        bias = _logic.pmv_bias(filtered)
        if bias.empty:
            st.info("No PMV/TSV pairs for these filters.")
        else:
            st.plotly_chart(_charts.pmv_bias_chart(bias), width="stretch")
            st.caption(
                "Positive bias: the ISO 7730 PMV stored in the database "
                "predicts warmer votes than occupants reported."
            )
