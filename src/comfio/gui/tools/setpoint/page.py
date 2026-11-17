"""Setpoint explorer — how far can the heating/cooling deadband widen?

Operative-temperature band edges are solved live from the calculator's
ASHRAE 55-2023 PMV. Energy savings are linked out to the CBE setpoint
energy calculator with the Hoyt et al. (2015) citation — no numeric
savings are shown (Open Decision 1: published values not supplied).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from comfio.gui.tools.calculator._models import ashrae_pmv
from comfio.gui.tools.setpoint import _logic

_T_AXIS = np.linspace(10.0, 36.0, 53)


def _curve(met: float, clo: float, vr: float, rh: float) -> np.ndarray:
    """PMV over the operative axis (tdb == tr)."""
    return ashrae_pmv(
        _T_AXIS,
        _T_AXIS,
        np.full_like(_T_AXIS, vr),
        np.full_like(_T_AXIS, rh),
        np.full_like(_T_AXIS, met),
        np.full_like(_T_AXIS, clo),
    )


def _band_chart(
    met: float,
    clo_heating: float,
    clo_cooling: float,
    rh: float,
    vr_still: float,
    vr_fan: float | None,
    band: dict[str, float],
    current_heat: float,
    current_cool: float,
    proposed_cool: float,
    pmv_limit: float,
) -> go.Figure:
    """PMV-vs-operative curves, ±limit band and setpoint markers."""
    fig = go.Figure()
    fig.add_hrect(
        y0=-pmv_limit,
        y1=pmv_limit,
        fillcolor="rgba(42,157,143,0.12)",
        line_width=0,
        annotation_text=f"±{pmv_limit:g} band",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=_T_AXIS,
            y=_curve(met, clo_heating, vr_still, rh),
            mode="lines",
            name=f"Heating clo {clo_heating:g} · {vr_still:g} m/s",
            line={"color": "#e76f51", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_T_AXIS,
            y=_curve(met, clo_cooling, vr_still, rh),
            mode="lines",
            name=f"Cooling clo {clo_cooling:g} · {vr_still:g} m/s",
            line={"color": "#457b9d", "width": 2},
        )
    )
    if vr_fan is not None:
        fig.add_trace(
            go.Scatter(
                x=_T_AXIS,
                y=_curve(met, clo_cooling, vr_fan, rh),
                mode="lines",
                name=f"Cooling clo {clo_cooling:g} · fan {vr_fan:g} m/s",
                line={"color": "#457b9d", "width": 2, "dash": "dash"},
            )
        )

    for x, label, color in (
        (current_heat, "current heating", "#e76f51"),
        (current_cool, "current cooling", "#457b9d"),
        (band["heat_min_c"], "proposed heating", "#b23a1d"),
        (proposed_cool, "proposed cooling", "#1d5a8a"),
    ):
        if np.isnan(x):
            continue
        dash = "solid" if label.startswith("current") else "dot"
        fig.add_vline(
            x=x,
            line_color=color,
            line_dash=dash,
            annotation_text=label,
            annotation_position="top",
        )
    fig.update_layout(
        height=420,
        xaxis_title="Operative temperature (tdb = tr) [°C]",
        yaxis_title="PMV",
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def render() -> None:
    st.title("Setpoint explorer")
    st.caption(
        "How far can the heating/cooling deadband widen while occupants "
        "stay inside the ASHRAE 55-2023 PMV ±0.5 band? Band edges are "
        "solved on the operative-temperature axis (tdb = tr); a missing "
        "edge is shown as outside model range, never extrapolated."
    )

    cols = st.columns(4)
    current_heat = float(
        cols[0].number_input(
            "Current heating setpoint [°C]",
            10.0,
            30.0,
            21.0,
            0.5,
            key="sp_heat",
        )
    )
    current_cool = float(
        cols[1].number_input(
            "Current cooling setpoint [°C]",
            10.0,
            30.0,
            25.0,
            0.5,
            key="sp_cool",
        )
    )
    met = float(cols[2].number_input("Metabolic rate [met]", 0.8, 4.0, 1.2, 0.1, key="sp_met"))
    rh = float(cols[3].number_input("Relative humidity [%]", 0.0, 100.0, 50.0, 5.0, key="sp_rh"))

    cols = st.columns(4)
    clo_heating = float(
        cols[0].number_input("Winter clothing [clo]", 0.0, 2.0, 1.0, 0.1, key="sp_clo_heat")
    )
    clo_cooling = float(
        cols[1].number_input("Summer clothing [clo]", 0.0, 2.0, 0.5, 0.1, key="sp_clo_cool")
    )
    vr_still = float(
        cols[2].number_input("Still-air speed [m/s]", 0.0, 1.0, 0.1, 0.05, key="sp_vr_still")
    )
    vr_fan_in = float(
        cols[3].number_input(
            "Fan air speed [m/s] (0 = none)",
            0.0,
            1.5,
            0.0,
            0.1,
            key="sp_vr_fan",
        )
    )
    vr_fan = vr_fan_in if vr_fan_in > 0.0 else None

    band = _logic.comfort_band(met, clo_heating, clo_cooling, rh, vr_still, vr_fan)
    proposed_cool = band["cool_max_fan_c"]
    if np.isnan(proposed_cool):
        proposed_cool = band["cool_max_c"]

    st.plotly_chart(
        _band_chart(
            met,
            clo_heating,
            clo_cooling,
            rh,
            vr_still,
            vr_fan,
            band,
            current_heat,
            current_cool,
            proposed_cool,
            0.5,
        ),
        width="stretch",
    )

    if np.isnan(band["heat_min_c"]) or np.isnan(proposed_cool):
        st.info(
            "At least one band edge is outside model range — no PMV ±0.5 "
            "crossing on 10–36 °C — so the proposed setpoint is NaN."
        )

    st.dataframe(
        _logic.deadband_table(current_heat, current_cool, band),
        hide_index=True,
        width="stretch",
    )

    with st.container(border=True):
        st.subheader("Energy impact")
        st.link_button("CBE setpoint energy savings calculator", _logic.ENERGY_CALC_URL)
        st.caption(
            "Setpoint-widening savings evidence: "
            f"{_logic.HOYT_2015}. PMV model: Tartarini & Schiavon (2020), "
            "doi:10.1016/j.softx.2020.100578."
        )
