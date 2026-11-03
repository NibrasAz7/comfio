"""Occupant survey — satisfaction-vote import and sensor-vs-occupant check.

Upload or generate a per-respondent satisfaction export (one row per
respondent, one column per category), remap categories to actual column
names, then review % satisfied per category — optionally grouped by a
zone/respondent column — and compare against the workbench's
sensor-domain scores when an analysis result exists in this session.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from comfio.gui import _charts, _data, _mapping, _state
from comfio.gui.tools.survey import _logic

_DF_KEY = "survey_df"


def _ingest() -> None:
    c1, c2 = st.columns([2, 1])
    uploaded = c1.file_uploader(
        "Upload survey results (.csv or .xlsx)",
        type=["csv", "xlsx", "xls"],
        help="One row per respondent; one satisfaction column per category.",
    )
    if c2.button("Load demo survey (60 respondents, 2 zones)"):
        st.session_state[_DF_KEY] = _logic.demo_survey()
        st.rerun()

    st.download_button(
        "Download survey template (CSV)",
        data=_logic.template_csv(),
        file_name="comfio_survey_template.csv",
        mime="text/csv",
    )

    if uploaded is None:
        return
    sheet: str | int = 0
    if _data.is_excel(uploaded.name):
        try:
            sheets = _data.excel_sheet_names(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not open {uploaded.name}: {exc}")
            return
        if len(sheets) > 1:
            sel = st.selectbox(
                "Workbook has multiple sheets — pick one",
                sheets,
                key="survey_excel_sheet",
            )
            sheet = sel if sel is not None else sheets[0]
    try:
        uploaded.seek(0)
        st.session_state[_DF_KEY] = _data.read_uploaded(uploaded, sheet=sheet)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {uploaded.name}: {exc}")


def _column_mapping(df: pd.DataFrame) -> dict[str, str]:
    """Per-category dropdown remap, seeded by ``detect_columns``."""
    st.subheader("1 · Column mapping")
    st.dataframe(df.head(5), width="stretch")
    auto = _logic.detect_columns(df)
    options = ["— not mapped —"] + [str(c) for c in df.columns]
    mapping: dict[str, str] = {}
    cols = st.columns(4)
    for i, canonical in enumerate(_logic.SAT_COLUMNS):
        pre = auto.get(canonical)
        idx = options.index(pre) if pre is not None and pre in options else 0
        chosen = cols[i % 4].selectbox(
            _logic.SAT_COLUMNS[canonical],
            options,
            index=idx,
            key=f"survey_map_{canonical}",
        )
        if chosen != "— not mapped —":
            mapping[canonical] = chosen
    return mapping


def _group_by(df: pd.DataFrame, mapping: dict[str, str]) -> str | None:
    """Optional ``by`` selector — defaults to a detected zone/respondent."""
    free = [str(c) for c in df.columns if c not in set(mapping.values())]
    options = ["— all respondents —"] + free
    preferred = next(
        (c for c in free if _mapping.normalize(c) in {"zone", "zone_id", "zone_name"}),
        None,
    )
    idx = options.index(preferred) if preferred is not None else 0
    chosen = st.selectbox(
        "Group results by (optional)",
        options,
        index=idx,
        key="survey_group_by",
    )
    return None if chosen is None or chosen == "— all respondents —" else chosen


def _scale(df: pd.DataFrame, mapping: dict[str, str]) -> Literal["-3..3", "1..7"] | None:
    """Validate every mapped column; offer the 1–7 remap only when seen."""
    invalid: list[tuple[str, str, list[str]]] = []
    any_seven = False
    for canonical, actual in mapping.items():
        status = _logic.validate_scale(df[actual])
        if status == "invalid":
            values = pd.to_numeric(df[actual], errors="coerce")
            non_response = df[actual].isna() | (df[actual].astype(str).str.strip() == "")
            bad = df[actual][values.isna() & ~non_response]
            samples = [str(v) for v in bad.unique()[:5]]
            if not samples:
                out = values[values.notna() & ~values.isin(range(-3, 8))]
                samples = [str(v) for v in out.unique()[:5]] or ["(empty)"]
            invalid.append((canonical, actual, samples))
        elif status == "1..7":
            any_seven = True
    if invalid:
        st.error(
            "These columns contain text answers or numbers outside the "
            "−3…3 / 1–7 scales — fix the file or remap the column:"
        )
        for canonical, actual, samples in invalid:
            st.write(
                f"• **{actual}** ({_logic.SAT_COLUMNS[canonical]}): "
                + ", ".join(samples)
            )
        return None
    if any_seven and st.checkbox(
        "Responses are on a 1–7 scale (remap to −3…3)",
        value=False,
        key="survey_scale_17",
    ):
        return "1..7"
    return "-3..3"


def _sensor_check(summary: pd.DataFrame) -> None:
    result = _state.get_result()
    if result is None:
        return
    st.subheader("3 · Sensor vs occupant")
    target = st.number_input(
        "Acceptability target [%]",
        min_value=1.0,
        max_value=100.0,
        value=_logic.THERMAL_ACCEPTABILITY_TARGET,
        step=1.0,
        help="User target; ASHRAE 55 defines 80 % for thermal only.",
    )
    comparison = _logic.sensor_vs_occupant(summary, result, target_pct=float(target))
    if comparison is None:
        st.info(
            "No shared domains — the survey needs at least one of Thermal, "
            "Air quality, Acoustics or Lighting, and the workbench run needs "
            "the matching sensor domain."
        )
        return
    st.dataframe(comparison, width="stretch")
    st.caption(
        "Sensor score = mean of the workbench domain score (0–100). "
        "Occupants pass when % satisfied ≥ target. "
        "Flagged rows are where the two sides disagree."
    )
    flagged = comparison[comparison["flag"]]
    if not flagged.empty:
        st.warning(
            "Sensors and occupants disagree on: "
            + ", ".join(f"{r.category} ({r.group})" for r in flagged.itertuples())
        )


def render() -> None:
    st.title("Occupant survey")
    st.caption(
        "Import per-respondent satisfaction votes (−3…+3 or 1–7), see "
        "% satisfied per category, and check the result against what the "
        "sensors measured."
    )
    _ingest()
    df = st.session_state.get(_DF_KEY)
    if df is None:
        st.info("Upload a survey file or load the demo survey to begin.")
        return
    assert isinstance(df, pd.DataFrame)

    mapping = _column_mapping(df)
    if not mapping:
        st.warning("Map at least one satisfaction column to continue.")
        return
    by = _group_by(df, mapping)
    scale = _scale(df, mapping)
    if scale is None:
        return

    summary = _logic.summarize(df, mapping, scale=scale, by=by)
    st.subheader("2 · Satisfaction summary")
    if summary.empty:
        st.info("No numeric answers in the mapped columns.")
        return
    st.dataframe(summary, width="stretch")
    st.plotly_chart(_charts.satisfaction_bars(summary), width="stretch")
    _sensor_check(summary)
