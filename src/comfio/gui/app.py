"""comfio — Streamlit analyst workbench.

Three-act flow:

    Act 1  INGEST   upload / demo → column mapper → capability banner
    Act 2  VERDICT  run_pipeline → IEQ gauge + domain tabs + assumptions
    Act 3  EVIDENCE compliance → CSV / DOCX / PDF / JSON / MD / script

Run via ``comfio.gui()`` or ``streamlit run src/comfio/gui/app.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from comfio.core.data_handler import SensorData
from comfio.gui import _charts, _data, _exports, _mapping, _params, _state
from comfio.performance.contracts import calculate_compliance
from comfio.reports.pipeline import detect_capabilities, run_pipeline

st.set_page_config(
    page_title="comfio — IEQ Analyst Workbench",
    page_icon="🌡️",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached engine call — reruns only when inputs change.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Running IEQ analysis…")
def _run(
    df: pd.DataFrame,
    column_map: dict[str, str],
    timestamp_col: str | None,
    config: dict,
) -> tuple[SensorData, Any]:
    sensor = SensorData(df=df, column_map=dict(column_map), timestamp_col=timestamp_col)
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    return sensor, run_pipeline(sensor, clean)


def _prepare(
    df: pd.DataFrame,
    mapping: dict[str, str],
    config: dict,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Inject assumption columns + apply NaN strategy. Returns notes."""
    df = df.copy()
    mapping = dict(mapping)
    notes: list[str] = []

    if _state.get(_state.IP_UNITS, False):
        df = _data.convert_ip_to_si(df, mapping)
        notes.append("Input declared in IP units — converted °F→°C and fpm→m/s.")

    if "radiant_temp_c" not in mapping and "air_temp_c" in mapping:
        if config.get("_mrt_equals_tdb", True):
            syn = "__comfio_radiant_temp_c"
            df[syn] = df[mapping["air_temp_c"]]
            mapping["radiant_temp_c"] = syn
            notes.append("Radiant temperature assumed equal to air temperature.")
        else:
            notes.append("No radiant temperature — thermal PMV unavailable.")

    if "air_velocity_ms" not in mapping:
        syn = "__comfio_air_velocity_ms"
        df[syn] = float(config.get("vr", 0.1))
        mapping["air_velocity_ms"] = syn
        notes.append(f"Air speed set to constant {config.get('vr', 0.1)} m/s.")

    if "outdoor_temp_c" not in mapping and "prevailing_mean_outdoor_c" not in mapping:
        notes.append("No outdoor temperature — adaptive thermal model skipped.")

    # NaN strategy on the mapped sensor columns
    strategy = _state.get(_state.NAN_STRATEGY, "interpolate")
    mapped_cols = [c for c in mapping.values() if c in df.columns]
    num = df[mapped_cols].apply(pd.to_numeric, errors="coerce")
    if strategy == "drop":
        df = df.loc[num.dropna().index].reset_index(drop=True)
    elif strategy == "fill_zero":
        df[mapped_cols] = num.fillna(0.0)
    else:  # interpolate
        df[mapped_cols] = num.interpolate().ffill().bfill()
    return df, mapping, notes


# --------------------------------------------------------------------------- #
# Act 1 — INGEST
# --------------------------------------------------------------------------- #
def act_ingest() -> None:
    st.title("comfio")
    st.caption(
        "Multi-domain Indoor Environmental Quality analysis — "
        "upload sensor data, get a standards-graded verdict and a defensible report."
    )

    c1, c2 = st.columns([2, 1])
    uploaded = c1.file_uploader(
        "Upload sensor data (.csv or .xlsx)",
        type=["csv", "xlsx", "xls"],
        help="One row per timestamp. Column names are auto-detected on the next step.",
    )
    if c2.button("Load demo data (1 week, all domains)"):
        _state.set_df(_data.demo_data())
        return

    with st.expander("Download a column template"):
        t1, t2 = st.columns(2)
        t1.download_button(
            "SI template (°C, m/s)",
            data=_data.template_csv("si"),
            file_name="comfio_template_si.csv",
            mime="text/csv",
        )
        t2.download_button(
            "IP template (°F, fpm)",
            data=_data.template_csv("ip"),
            file_name="comfio_template_ip.csv",
            mime="text/csv",
        )

    if uploaded is not None:
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
                    key="excel_sheet",
                )
                sheet = sel if sel is not None else sheets[0]
        try:
            uploaded.seek(0)
            _state.set_df(_data.read_uploaded(uploaded, sheet=sheet))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read {uploaded.name}: {exc}")


# --------------------------------------------------------------------------- #
# Act 1.5 — column mapping + capability banner
# --------------------------------------------------------------------------- #
def act_mapping(df: pd.DataFrame) -> None:
    st.subheader("1 · Column mapping")
    st.dataframe(df.head(5), width="stretch")

    auto = _state.get_column_map() or _mapping.auto_map_columns(df)
    timestamp_col = _state.get_timestamp_col()
    sensor_cols = [
        c for c in _mapping.CANONICAL_COLUMNS if c in _data.COLUMN_DESCRIPTIONS
    ]
    options = ["— not available —"] + list(df.columns)

    ts_idx = options.index(timestamp_col) if timestamp_col in options else 0
    timestamp_sel = st.selectbox(
        "Timestamp (optional, enables 'when did it fail' analysis)",
        options,
        index=ts_idx,
    )

    with st.form("column_mapping"):
        st.markdown("Map your file's columns to comfio parameters.")
        cols = st.columns(3)
        mapping: dict[str, str] = {}
        for i, canonical in enumerate(sensor_cols):
            pre = auto.get(canonical)
            idx = options.index(pre) if pre in options else 0
            label = _data.COLUMN_DESCRIPTIONS.get(canonical, canonical)
            chosen = cols[i % 3].selectbox(
                label, options, index=idx, key=f"map_{canonical}"
            )
            if chosen is not None and chosen != "— not available —":
                mapping[canonical] = chosen
        submitted = st.form_submit_button("Run analysis", type="primary")

    preview = dict(mapping)
    if "radiant_temp_c" not in preview and "air_temp_c" in preview:
        preview["radiant_temp_c"] = "(assumed = air temp)"
    if "air_velocity_ms" not in preview:
        preview["air_velocity_ms"] = "(constant)"
    coverage = _mapping.domain_coverage(preview)
    parts = [f"**{d.capitalize()}** {'✅' if ok else '—'}" for d, ok in coverage.items()]
    missing = ", ".join(d for d, ok in coverage.items() if not ok)
    st.info(
        "Domain coverage from this file: "
        + " · ".join(parts)
        + (f" — no data for: {missing}" if missing else "")
    )

    unm = _mapping.unmapped_columns(df, mapping)
    if unm:
        st.caption(f"Unmapped columns (ignored): {', '.join(unm)}")

    if st.button("Upload a different file"):
        _state.reset()
        st.rerun()

    has_outdoor = any(
        k in auto
        for k in ("outdoor_temp_c", "prevailing_mean_outdoor_c", "running_mean_outdoor_c")
    )
    if (
        not has_outdoor
        and timestamp_sel is not None
        and timestamp_sel != "— not available —"
    ):
        with st.expander("No outdoor temperature — fetch from meteostat?"):
            st.caption(
                "Adds an outdoor_temp_c column (hourly, nearest match) so the "
                "adaptive thermal model can run. Requires network access."
            )
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", value=40.0, format="%.4f")
            lon = c2.number_input("Longitude", value=-74.0, format="%.4f")
            if st.button("Fetch outdoor temperature for this file's range"):
                try:
                    with st.spinner("Fetching weather data…"):
                        df2 = _data.attach_outdoor_temperature(
                            df, timestamp_sel, lat, lon
                        )
                    _state.set_df(df2)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Weather fetch failed: {exc}")

    if submitted:
        if not any(coverage.values()):
            st.error(
                "No evaluable domain — map at least air temperature + humidity, "
                "CO₂, illuminance, or sound level."
            )
            return
        _state.set(
            _state.TIMESTAMP_COL,
            timestamp_sel if timestamp_sel != "— not available —" else None,
        )
        _state.set_column_map(mapping)


# --------------------------------------------------------------------------- #
# Act 2 — VERDICT
# --------------------------------------------------------------------------- #
def act_verdict(
    sensor: SensorData, result: Any, df: pd.DataFrame, notes: list[str]
) -> None:
    st.subheader("2 · Verdict")
    ieq = result.ieq_result
    if ieq is None:
        st.warning("No domain could be evaluated — check the column mapping.")
        return

    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(
        _charts.gauge(float(np.nanmean(ieq.index))), width="stretch"
    )
    c2.plotly_chart(_charts.domain_bars(ieq.domain_scores), width="stretch")

    ts_series = sensor.get_timestamps()
    x_axis = (
        pd.to_datetime(ts_series, errors="coerce")
        if ts_series is not None
        else pd.Series(np.arange(len(df)))
    )
    has_time = ts_series is not None and not x_axis.isna().all()

    names = list(ieq.domain_scores)
    tabs = st.tabs([n.capitalize() for n in names] + ["Timeline"])
    for tab, domain in zip(tabs[:-1], names, strict=True):
        with tab:
            scores = ieq.domain_scores[domain]
            st.plotly_chart(
                _charts.timeseries(
                    x_axis,
                    scores,
                    f"{domain} score",
                    _charts.DOMAIN_COLORS.get(domain, "#888"),
                    threshold=80.0,
                ),
                width="stretch",
            )
            if domain == "thermal" and "thermal" in result.domain_results:
                tr = result.domain_results["thermal"]
                tdb = sensor.get_column("air_temp_c")
                rh = sensor.get_column("relative_humidity_pct")
                st.plotly_chart(
                    _charts.comfort_scatter(tdb, rh, np.asarray(tr.pmv)),
                    width="stretch",
                )
                s1, s2, s3 = st.columns(3)
                s1.metric("Mean PMV", f"{np.nanmean(tr.pmv):.2f}")
                s2.metric("Mean PPD", f"{np.nanmean(tr.ppd):.1f} %")
                s3.metric(
                    "Samples |PMV|>0.5",
                    f"{int(np.nansum(np.abs(tr.pmv) > 0.5))} / {len(tr.pmv)}",
                )
            if domain == "tsv" and "tsv" in result.domain_results:
                tsv_res = result.domain_results["tsv"]
                st.metric("Mean TSV", f"{tsv_res.mean_tsv:.2f}")

    with tabs[-1]:
        if has_time:
            st.plotly_chart(
                _charts.exceedance_heatmap(
                    pd.Series(x_axis.to_numpy()), np.asarray(ieq.index)
                ),
                width="stretch",
            )
            st.caption("When the Global IEQ Index falls below the 80 threshold.")
        else:
            st.plotly_chart(
                _charts.timeseries(
                    x_axis, np.asarray(ieq.index), "IEQ index", "#264653", 80.0
                ),
                width="stretch",
            )
            st.caption("No timestamp column — index over sample order.")

    with st.expander("Assumptions & configuration used", expanded=True):
        for n in notes:
            st.write("•", n)
        for w in result.warnings:
            st.write("•", w)
        st.json(result.config)

    # Occupant-vote calibration: PMV → TSV regression when both exist.
    if "tsv" in result.domain_results and "thermal" in result.domain_results:
        tsv_res = result.domain_results["tsv"]
        votes = np.asarray(
            result.domain_results.get("tsv_augmented", tsv_res.tsv), dtype=float
        )
        pmv = np.asarray(result.domain_results["thermal"].pmv, dtype=float)
        mask = ~(np.isnan(votes) | np.isnan(pmv)) if len(votes) == len(pmv) else None
        if mask is not None and int(mask.sum()) >= 10:
            from comfio.domains.thermal_personal import train_personalisation

            idx = train_personalisation(pmv[mask], votes[mask])
            with st.expander("Occupant-vote calibration (PMV → TSV)"):
                p1, p2, p3 = st.columns(3)
                p1.metric("Slope (α)", f"{idx.alpha:.3f}")
                p2.metric("Offset (β)", f"{idx.beta:.3f}")
                p3.metric("R²", f"{idx.r_squared:.3f}")
                st.plotly_chart(
                    _charts.pmv_tsv_scatter(
                        pmv[mask], votes[mask], idx.alpha, idx.beta
                    ),
                    width="stretch",
                )
                if "tsv_augmented" in result.domain_results:
                    st.caption(
                        "Sparse votes were augmented onto the full timeline "
                        "(CDF remapping) before fitting."
                    )

    with st.expander("Pin this run as a scenario"):
        n_pin = len(_state.get_scenarios()) + 1
        pin_name = st.text_input("Scenario name", value=f"Scenario {n_pin}")
        if st.button("Pin scenario"):
            comp = getattr(result, "compliance_report", None)
            _state.pin_scenario(
                {
                    "name": pin_name or f"Scenario {n_pin}",
                    "ieq_mean": float(np.nanmean(ieq.index)),
                    "domain_means": {
                        d: float(np.nanmean(s))
                        for d, s in ieq.domain_scores.items()
                    },
                    "compliance_rate_pct": (
                        comp.compliance_rate_pct if comp is not None else None
                    ),
                    "config": dict(result.config),
                }
            )
            st.rerun()


# --------------------------------------------------------------------------- #
# Act 2.5 — SCENARIOS
# --------------------------------------------------------------------------- #
def act_scenarios() -> None:
    scenarios = _state.get_scenarios()
    if not scenarios:
        return
    st.subheader("Scenario comparison")
    if len(scenarios) == 1:
        st.caption(
            f"'{scenarios[0]['name']}' pinned — run another file or "
            "configuration and pin it to compare."
        )
        return

    domains = sorted({d for s in scenarios for d in s["domain_means"]})
    rows = [
        {
            "scenario": s["name"],
            "IEQ mean": round(s["ieq_mean"], 1),
            **{d: round(s["domain_means"].get(d, np.nan), 1) for d in domains},
            "compliance %": (
                round(s["compliance_rate_pct"], 1)
                if s.get("compliance_rate_pct") is not None
                else None
            ),
        }
        for s in scenarios
    ]
    st.dataframe(pd.DataFrame(rows).set_index("scenario"), width="stretch")
    st.plotly_chart(_charts.scenario_bars(scenarios), width="stretch")
    if st.button("Clear scenarios"):
        _state.clear_scenarios()
        st.rerun()


# --------------------------------------------------------------------------- #
# Act 3 — EVIDENCE
# --------------------------------------------------------------------------- #
def act_evidence(result: Any, config: dict) -> None:
    st.subheader("3 · Evidence")
    ieq = result.ieq_result
    if ieq is None:
        return

    threshold, zone_id = _params.render_evidence_params()
    report = calculate_compliance(
        ieq,
        threshold=threshold,
        domain_compliant_arrays={
            d: r.compliant
            for d, r in result.domain_results.items()
            if getattr(r, "compliant", None) is not None
        },
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Compliance rate", f"{report.compliance_rate_pct:.1f} %")
    m2.metric("IEQ avg", f"{report.ieq_index_avg:.1f}")
    m3.metric("IEQ min", f"{report.ieq_index_min:.1f}")
    m4.metric(
        "Non-compliant hours",
        f"{report.total_hours - report.compliant_hours:.0f} / {report.total_hours:.0f} h",
    )

    std_labels = {
        "thermal": f"ISO 7730 cat-{config.get('category')}",
        "spmv": "sPMV (advisory)",
        "adaptive": "Adaptive comfort model",
        "visual": f"EN 12464-1 · {config.get('task_type')}",
        "acoustic": str(config.get("nc_level")),
        "iaq": f"CO₂ {config.get('co2_threshold')}",
        "pollutant_iaq": "WHO pollutant guideline",
        "tsv": "|TSV| acceptability band",
    }
    rows = [
        {
            "domain": d,
            "standard / level": std_labels.get(d, d),
            "samples passing": f"{int(np.nansum(r.compliant))} / {len(r.compliant)}",
            "pass rate %": round(
                100.0 * float(np.nanmean(np.asarray(r.compliant, dtype=float))), 1
            ),
        }
        for d, r in result.domain_results.items()
        if getattr(r, "compliant", None) is not None
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("domain"), width="stretch")

    st.caption(
        f"Standards: ISO 7730 cat-{config.get('category')} · EN 12464 "
        f"{config.get('task_type')} · {config.get('nc_level')} · "
        f"CO₂ {config.get('co2_threshold')} · {config.get('weight_preset')} weights · "
        f"n={ieq.n_timestamps} samples"
    )

    _exports.render_exports(ieq, report, zone_id, config)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    df = _state.get_df()
    if df is None:
        df = _data.load_preloaded()
        if df is not None:
            _state.set_df(df)

    if df is None:
        act_ingest()
        return

    assert df is not None
    config = _params.render_sidebar(set(df.columns))

    if not _state.is_mapped():
        act_mapping(df)
        return

    prepared, mapping, notes = _prepare(
        df, _state.get_column_map() or {}, config
    )
    ts_col = _state.get_timestamp_col()
    sensor, result = _run(prepared, mapping, ts_col, config)

    st.title("comfio — analysis")
    caps = detect_capabilities(sensor)
    banner = " · ".join(
        f"{k.replace('_', ' ')} {'✅' if v else '—'}" for k, v in caps.items() if v
    )
    st.info(f"Ran: {banner}")
    if st.button("← Back to column mapping"):
        _state.clear_mapping()
        st.rerun()

    act_verdict(sensor, result, prepared, notes)
    act_scenarios()
    st.divider()
    act_evidence(result, config)


main()
