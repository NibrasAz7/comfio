"""Comfort calculator — CBE Comfort Tool parity.

Three tabs: **Result** (metrics + psychrometric/adaptive chart + local
discomfort), **Compare** (up to 3 input sets on one chart), **Ranges**
(sweep one input over PMV/SET/PPD curves).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comfio.gui import _charts, _state
from comfio.gui.tools.calculator._models import (
    ADAPTIVE_METHODS,
    METHODS,
    Inputs,
    PointResult,
    evaluate_point,
    mrt_from_globe,
    rh_from,
    sweep,
)
from comfio.gui.tools.calculator._psychro import humidity_ratio, pmv_zone_boundary

_HumKind = Literal["rh", "dew_point", "humidity_ratio", "wet_bulb"]
_HUM_KINDS: dict[str, _HumKind] = {
    "Relative humidity [%]": "rh",
    "Dew point [°C]": "dew_point",
    "Humidity ratio [kg/kg]": "humidity_ratio",
    "Wet bulb [°C]": "wet_bulb",
}
_MRT_MODES = ("= air temperature", "measured", "from globe temperature")
_CUSTOM = "Custom…"
_PARAM_LABELS = {
    "tdb": "Air temperature [°C]",
    "tr": "MRT [°C]",
    "rh": "Relative humidity [%]",
    "vr": "Air speed [m/s]",
    "met": "Metabolic rate [met]",
    "clo": "Clothing [clo]",
}


@st.cache_data(show_spinner="Computing comfort zone…")
def _zone(
    met: float,
    clo: float,
    vr: float,
    dynamic_clo: bool,
    relative_air_speed: bool,
) -> pd.DataFrame:
    return pmv_zone_boundary(
        met,
        clo,
        vr,
        dynamic_clo=dynamic_clo,
        relative_air_speed=relative_air_speed,
    )


def _fmt(x: float, spec: str = "+.2f") -> str:
    return "outside model range" if np.isnan(x) else f"{x:{spec}}"


def _pick(
    label: str,
    options: dict[str, float],
    default_label: str,
    custom_default: float,
    custom_min: float,
    custom_max: float,
    key: str,
) -> float:
    """Named-value selectbox plus a 'Custom…' number input."""
    names = list(options) + [_CUSTOM]
    sel = st.selectbox(label, names, index=names.index(default_label), key=f"calc_{key}_pick")
    if sel == _CUSTOM:
        return float(
            st.number_input(
                f"{label} (custom)",
                min_value=custom_min,
                max_value=custom_max,
                value=custom_default,
                key=f"calc_{key}_custom",
            )
        )
    return float(options[sel])


def _prefill_picker(
    value: float,
    options: dict[str, float],
    key: str,
) -> None:
    """Prefill a select+custom pair to represent ``value``."""
    for name, v in options.items():
        if abs(v - value) < 1e-9:
            st.session_state[f"calc_{key}_pick"] = name
            return
    st.session_state[f"calc_{key}_pick"] = _CUSTOM
    st.session_state[f"calc_{key}_custom"] = float(value)


def _prefill() -> None:
    """Consume CALC_INPUTS once, writing widget keys before they exist."""
    data = _state.get(_state.CALC_INPUTS)
    if not isinstance(data, dict):
        return
    st.session_state.pop(_state.CALC_INPUTS, None)
    if "tdb" in data:
        st.session_state["calc_tdb"] = float(data["tdb"])
    if "tr" in data:
        st.session_state["calc_mrt_mode"] = "measured"
        st.session_state["calc_tr"] = float(data["tr"])
    if "rh" in data:
        st.session_state["calc_hum_kind"] = "Relative humidity [%]"
        st.session_state["calc_hum_value_rh"] = float(data["rh"])
    if "vr" in data:
        st.session_state["calc_vr"] = float(data["vr"])
    if "met" in data:
        from pythermalcomfort.utilities import met_typical_tasks

        _prefill_picker(float(data["met"]), met_typical_tasks, "met")
    if "clo" in data:
        from pythermalcomfort.utilities import clo_typical_ensembles

        _prefill_picker(float(data["clo"]), clo_typical_ensembles, "clo")
    if "t_running_mean" in data:
        st.session_state["calc_trm"] = float(data["t_running_mean"])


def _inputs(method: str) -> tuple[Inputs, bool, bool]:
    """Render the input widgets and build the current ``Inputs``."""
    from pythermalcomfort.utilities import clo_typical_ensembles, met_typical_tasks

    is_pmv = method not in ADAPTIVE_METHODS
    c1, c2 = st.columns(2)
    tdb = float(c1.number_input("Air temperature [°C]", -30.0, 60.0, 24.0, 0.5, key="calc_tdb"))
    mrt_mode = c2.radio("MRT", _MRT_MODES, horizontal=True, key="calc_mrt_mode")
    if mrt_mode == "measured":
        tr = float(c2.number_input("MRT [°C]", -30.0, 80.0, 24.0, 0.5, key="calc_tr"))
    elif mrt_mode == "from globe temperature":
        g1, g2 = c2.columns(2)
        tg = float(g1.number_input("Globe temperature [°C]", -30.0, 80.0, 24.0, 0.5, key="calc_tg"))
        d = float(g2.number_input("Globe diameter [m]", 0.02, 0.30, 0.15, 0.01, key="calc_globe_d"))
        vr_now = float(st.session_state.get("calc_vr", 0.1))
        tr = mrt_from_globe(tg, tdb, vr_now, d)
    else:
        tr = tdb

    hum_kind = c1.selectbox("Humidity input", list(_HUM_KINDS), key="calc_hum_kind")
    kind = _HUM_KINDS[hum_kind]
    hum_defaults = {
        "rh": (0.0, 100.0, 50.0, 1.0, "RH [%]"),
        "dew_point": (-50.0, 50.0, 10.0, 0.5, "Dew point [°C]"),
        "humidity_ratio": (0.0, 0.05, 0.008, 0.001, "Humidity ratio [kg/kg]"),
        "wet_bulb": (-50.0, 50.0, 15.0, 0.5, "Wet bulb [°C]"),
    }
    lo, hi, dflt, step, hum_label = hum_defaults[kind]
    hum_value = float(c1.number_input(hum_label, lo, hi, dflt, step, key=f"calc_hum_value_{kind}"))
    rh = rh_from(kind, hum_value, tdb)
    if np.isnan(rh):
        st.warning(
            f"Could not derive relative humidity from {hum_label.lower()} "
            f"{hum_value} — no solution at tdb {tdb} °C."
        )

    vr = float(c2.number_input("Air speed [m/s]", 0.0, 3.0, 0.10, 0.05, key="calc_vr"))

    met = _pick(
        "Metabolic rate [met]",
        met_typical_tasks,
        "Standing, relaxed",
        1.2,
        0.7,
        8.0,
        "met",
    )
    clo = _pick(
        "Clothing [clo]",
        clo_typical_ensembles,
        "Typical summer indoor clothing",
        0.5,
        0.0,
        2.5,
        "clo",
    )

    trm = None
    if not is_pmv:
        trm_label = (
            "Prevailing mean outdoor temperature [°C]"
            if method == ADAPTIVE_METHODS[0]
            else "Running mean outdoor temperature [°C]"
        )
        trm = float(st.number_input(trm_label, -30.0, 60.0, 20.0, 0.5, key="calc_trm"))

    dynamic_clo = True
    relative_air_speed = True
    if is_pmv:
        dynamic_clo = bool(
            st.checkbox(
                "Dynamic clothing insulation (ASHRAE 55)",
                value=True,
                key="calc_dynamic_clo",
                help="Corrects clo for body movement per the 55-2023 model.",
            )
        )
        relative_air_speed = bool(
            st.checkbox(
                "Relative air speed (ASHRAE 55)",
                value=True,
                key="calc_relative_vr",
                help="Adjusts air speed for metabolic rate above 1.0 met.",
            )
        )

    return (
        Inputs(tdb=tdb, tr=tr, rh=rh, vr=vr, met=met, clo=clo, t_running_mean=trm),
        dynamic_clo,
        relative_air_speed,
    )


def _point_xy(inp: Inputs) -> tuple[float, float]:
    """(operative temperature, humidity ratio) for the psychro chart."""
    from pythermalcomfort.utilities import operative_tmp

    t_op = float(operative_tmp(np.array([inp.tdb]), np.array([inp.tr]), np.array([inp.vr]))[0])
    w = float(humidity_ratio(np.array([inp.tdb]), np.array([inp.rh]))[0])
    return t_op, w


def _metrics(res: PointResult) -> None:
    compliant = "—" if res.compliant is None else "yes" if res.compliant else "no"
    if res.method in ADAPTIVE_METHODS:
        lo = res.extra.get("lower_80", res.extra.get("lower_ii", np.nan))
        hi = res.extra.get("upper_80", res.extra.get("upper_ii", np.nan))
        cols = st.columns(5)
        cols[0].metric("Operative t [°C]", _fmt(res.extra.get("t_op", np.nan), ".1f"))
        cols[1].metric("Comfort t [°C]", _fmt(res.extra.get("t_comf", np.nan), ".1f"))
        cols[2].metric(
            "Band [°C]",
            "outside model range" if np.isnan(lo) else f"{lo:.1f}–{hi:.1f}",
        )
        cols[3].metric("Category", res.category or "—")
        cols[4].metric("Compliant", compliant)
        return
    cols = st.columns(6)
    cols[0].metric("PMV", _fmt(res.pmv))
    cols[1].metric("PPD", _fmt(res.ppd, ".1f"))
    cols[2].metric("SET [°C]", _fmt(res.set_c, ".1f"))
    cols[3].metric("CE [°C]", _fmt(res.ce, ".1f"))
    cols[4].metric("Category", res.category or "—")
    cols[5].metric("Compliant", compliant)


def _local_discomfort(inp: Inputs) -> None:
    from comfio.domains.thermal_local import (
        evaluate_ankle_draft,
        evaluate_vertical_gradient,
    )

    with st.expander("Local discomfort (ASHRAE 55)"):
        c1, c2 = st.columns(2)
        v_ankle = float(
            c1.number_input(
                "Air speed at ankle (0.1 m) [m/s]",
                0.0,
                1.0,
                min(inp.vr, 0.2),
                0.01,
                key="calc_v_ankle",
            )
        )
        grad = float(
            c2.number_input(
                "Head–ankle temperature difference [°C]",
                0.0,
                15.0,
                2.0,
                0.5,
                key="calc_vgrad",
            )
        )
        ad = evaluate_ankle_draft(
            tdb=np.array([inp.tdb]),
            tr=np.array([inp.tr]),
            vr=np.array([inp.vr]),
            rh=np.array([inp.rh]),
            met=inp.met,
            clo=inp.clo,
            v_ankle=v_ankle,
        )
        vg = evaluate_vertical_gradient(
            tdb=np.array([inp.tdb]),
            tr=np.array([inp.tr]),
            vr=np.array([inp.vr]),
            rh=np.array([inp.rh]),
            met=inp.met,
            clo=inp.clo,
            vertical_tmp_grad=grad,
        )
        m1, m2 = st.columns(2)
        m1.metric("Ankle draft PPD", _fmt(float(ad.ppd_ad[0]), ".1f"))
        m2.metric("Vertical-gradient PPD", _fmt(float(vg.ppd_vg[0]), ".1f"))
        st.caption(
            "NaN means the inputs fall outside the local-discomfort model "
            "domains (ankle draft: tdb 20–26 °C; gradient: standing occupant)."
        )


def _compare_table(results: list[PointResult]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append(
            {
                "set": f"Set {i}",
                "PMV": r.pmv,
                "PPD": r.ppd,
                "SET": r.set_c,
                "CE": r.ce,
                "category": r.category,
                "compliant": r.compliant,
            }
        )
    return pd.DataFrame(rows).set_index("set")


def render() -> None:
    _prefill()
    st.title("Comfort calculator")
    st.caption(
        "Point thermal comfort per ASHRAE 55-2023, ISO 7730:2005 and "
        "EN 16798-1:2019 — PMV/PPD, SET, cooling effect, adaptive models, "
        "local discomfort."
    )

    method = st.selectbox("Method", METHODS, key="calc_method")
    inp, dynamic_clo, relative_vr = _inputs(method)
    res = evaluate_point(
        inp,
        method,
        dynamic_clo=dynamic_clo,
        relative_air_speed=relative_vr,
    )
    is_pmv = method not in ADAPTIVE_METHODS

    tab_result, tab_compare, tab_ranges = st.tabs(["Result", "Compare", "Ranges"])

    with tab_result:
        _metrics(res)
        for note in res.notes:
            st.warning(note)
        if is_pmv:
            zone = _zone(inp.met, inp.clo, inp.vr, dynamic_clo, relative_vr)
            t_op, w = _point_xy(inp)
            st.plotly_chart(
                _charts.psychro_chart([("PMV ±0.5 zone", zone)], [("current point", t_op, w)]),
                width="stretch",
            )
            _local_discomfort(inp)
        else:
            point = (
                (inp.t_running_mean, res.extra["t_op"])
                if inp.t_running_mean is not None and "t_op" in res.extra
                else None
            )
            st.plotly_chart(
                _charts.adaptive_chart("ashrae" if method == ADAPTIVE_METHODS[0] else "en", point),
                width="stretch",
            )
        st.caption(
            "SET and CE are ASHRAE 55 metrics computed with the same "
            "dynamic-clo/relative-air-speed preprocessing. EN 16798-1 "
            "compliance is Category II (normal expectation)."
        )

    with tab_compare:
        cols = st.columns(3)
        cmp_inputs: list[Inputs] = []
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"**Set {i + 1}**")
                cmp_inputs.append(
                    Inputs(
                        tdb=float(
                            st.number_input(
                                "Air t [°C]",
                                -30.0,
                                60.0,
                                inp.tdb,
                                0.5,
                                key=f"calc_cmp{i}_tdb",
                            )
                        ),
                        tr=float(
                            st.number_input(
                                "MRT [°C]",
                                -30.0,
                                80.0,
                                inp.tr,
                                0.5,
                                key=f"calc_cmp{i}_tr",
                            )
                        ),
                        rh=float(
                            st.number_input(
                                "RH [%]",
                                0.0,
                                100.0,
                                min(max(inp.rh, 0.0), 100.0) if np.isfinite(inp.rh) else 50.0,
                                1.0,
                                key=f"calc_cmp{i}_rh",
                            )
                        ),
                        vr=float(
                            st.number_input(
                                "v [m/s]",
                                0.0,
                                3.0,
                                inp.vr,
                                0.05,
                                key=f"calc_cmp{i}_vr",
                            )
                        ),
                        met=float(
                            st.number_input(
                                "met",
                                0.7,
                                8.0,
                                inp.met,
                                0.1,
                                key=f"calc_cmp{i}_met",
                            )
                        ),
                        clo=float(
                            st.number_input(
                                "clo",
                                0.0,
                                2.5,
                                inp.clo,
                                0.05,
                                key=f"calc_cmp{i}_clo",
                            )
                        ),
                        t_running_mean=(
                            float(
                                st.number_input(
                                    "Outdoor t [°C]",
                                    -30.0,
                                    60.0,
                                    inp.t_running_mean if inp.t_running_mean is not None else 20.0,
                                    0.5,
                                    key=f"calc_cmp{i}_trm",
                                )
                            )
                            if not is_pmv
                            else None
                        ),
                    )
                )
        cmp_results = [
            evaluate_point(
                ci,
                method,
                dynamic_clo=dynamic_clo,
                relative_air_speed=relative_vr,
            )
            for ci in cmp_inputs
        ]
        st.dataframe(_compare_table(cmp_results), width="stretch")
        if is_pmv:
            zones = []
            pts = []
            for i, ci in enumerate(cmp_inputs):
                zones.append(
                    (f"Set {i + 1}", _zone(ci.met, ci.clo, ci.vr, dynamic_clo, relative_vr))
                )
                t_op, w = _point_xy(ci)
                pts.append((f"Set {i + 1}", t_op, w))
            st.plotly_chart(_charts.psychro_chart(zones, pts), width="stretch")
        else:
            fig = _charts.adaptive_chart("ashrae" if method == ADAPTIVE_METHODS[0] else "en", None)
            for i, (ci, cr) in enumerate(zip(cmp_inputs, cmp_results, strict=True)):
                if ci.t_running_mean is not None and "t_op" in cr.extra:
                    fig.add_trace(
                        go.Scatter(
                            x=[ci.t_running_mean],
                            y=[cr.extra["t_op"]],
                            mode="markers",
                            marker={"symbol": "star", "size": 13},
                            name=f"Set {i + 1}",
                        )
                    )
            st.plotly_chart(fig, width="stretch")

    with tab_ranges:
        if not is_pmv:
            st.info(
                "Ranges sweeps PMV/SET/PPD and is available for the PMV "
                "methods — the adaptive models have no PMV axis."
            )
        else:
            params = ["tdb", "tr", "rh", "vr", "met", "clo"]
            param = st.selectbox(
                "Parameter",
                params,
                format_func=lambda p: _PARAM_LABELS[p],
                key="calc_rng_param",
            )
            cur = getattr(inp, param)
            defaults = {
                "tdb": (cur - 5.0, cur + 5.0),
                "tr": (cur - 5.0, cur + 5.0),
                "rh": (0.0, 100.0),
                "vr": (0.0, 2.0),
                "met": (0.8, 4.0),
                "clo": (0.0, 2.0),
            }
            d_lo, d_hi = defaults[param]
            r1, r2, r3 = st.columns(3)
            lo = float(r1.number_input("from", value=d_lo, key="calc_rng_lo"))
            hi = float(r2.number_input("to", value=d_hi, key="calc_rng_hi"))
            n_pts = int(r3.slider("points", 3, 41, 21, key="calc_rng_n"))
            if lo >= hi:
                st.error("'from' must be below 'to'.")
            else:
                df = sweep(inp, method, param, np.linspace(lo, hi, n_pts))
                st.plotly_chart(
                    _charts.ranges_curve(df, param, _PARAM_LABELS[param]),
                    width="stretch",
                )

    st.caption(
        "Sources: pythermalcomfort (Tartarini & Schiavon 2020, "
        "doi:10.1016/j.softx.2020.100578) · ASHRAE 55-2023 · "
        "ISO 7730:2005 · EN 16798-1:2019. All values are computed live; "
        "SI units."
    )
