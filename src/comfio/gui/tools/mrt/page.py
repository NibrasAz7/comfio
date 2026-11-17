"""MRT room designer — plan-view MRT/PMV map of a rectangular room.

Surface temperatures are uniform per enclosure surface; MRT comes from
Hottel view factors (``comfio.domains.thermal_mrt``), the PMV map reuses
the pipeline's ``evaluate_thermal`` with uniform air/humidity inputs.
Optional short-wave solar ΔMRT (ASHRAE 55 §C2) is applied at the
occupant point only — direct-beam sun-patch geometry is out of scope.
"""

from __future__ import annotations

import warnings

import numpy as np
import streamlit as st

from comfio.domains.thermal import evaluate_thermal
from comfio.domains.thermal_mrt import (
    SURFACES,
    Room,
    Window,
    mrt_grid,
    mrt_point,
    solar_delta_mrt,
)
from comfio.gui import _charts, _state

_N_WINDOWS = 2
_SPACING = 0.25
_MARGIN = 0.25


@st.cache_data(show_spinner="Computing MRT grid…")
def _cached_grid(
    room: Room, posture: str, spacing: float, margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return mrt_grid(
        room,
        posture=posture,  # type: ignore[arg-type]
        spacing=spacing,
        margin=margin,
    )


def _geometry_inputs() -> tuple[float, float, float]:
    cols = st.columns(3)
    length = float(
        cols[0].number_input("Room length L [m] (x)", 0.5, 30.0, 6.0, 0.1, key="mrt_len")
    )
    width = float(cols[1].number_input("Room width W [m] (y)", 0.5, 30.0, 4.0, 0.1, key="mrt_wid"))
    height = float(cols[2].number_input("Room height H [m] (z)", 2.0, 8.0, 3.0, 0.1, key="mrt_hei"))
    return length, width, height


def _temps_inputs() -> dict[str, float]:
    labels = {
        "front": "Front (x=0) [°C]",
        "rear": "Rear (x=L) [°C]",
        "left": "Left (y=0) [°C]",
        "right": "Right (y=W) [°C]",
        "floor": "Floor [°C]",
        "ceiling": "Ceiling [°C]",
    }
    cols = st.columns(len(SURFACES))
    temps: dict[str, float] = {}
    for i, surf in enumerate(SURFACES):
        temps[surf] = float(
            cols[i].number_input(labels[surf], -20.0, 80.0, 22.0, 0.5, key=f"mrt_t_{surf}")
        )
    return temps


def _window_inputs() -> list[Window]:
    windows: list[Window] = []
    cols = st.columns(_N_WINDOWS)
    for i in range(_N_WINDOWS):
        with cols[i]:
            active = st.checkbox(f"Window {i + 1} active", value=i == 0, key=f"mrt_w{i}_on")
            if not active:
                continue
            wall = st.selectbox(
                "Wall",
                ("front", "rear", "left", "right"),
                key=f"mrt_w{i}_wall",
                help="u0 is measured along the wall: from the y=0 corner "
                "for front/rear, from the x=0 corner for left/right.",
            )
            c1, c2 = st.columns(2)
            u0 = float(
                c1.number_input("u0 along wall [m]", 0.0, 30.0, 1.0, 0.1, key=f"mrt_w{i}_u0")
            )
            z0 = float(
                c2.number_input("Sill height z0 [m]", 0.0, 7.0, 0.9, 0.1, key=f"mrt_w{i}_z0")
            )
            w = float(c1.number_input("Width [m]", 0.1, 10.0, 2.0, 0.1, key=f"mrt_w{i}_w"))
            h = float(c2.number_input("Height [m]", 0.1, 7.0, 1.4, 0.1, key=f"mrt_w{i}_h"))
            temp = float(
                st.number_input(
                    "Glazing temperature [°C]",
                    -20.0,
                    80.0,
                    18.0,
                    0.5,
                    key=f"mrt_w{i}_t",
                )
            )
            windows.append(
                Window(wall=wall, u0=u0, z0=z0, width=w, height=h, temp_c=temp)  # type: ignore[arg-type]
            )
    return windows


def _pmv_inputs() -> dict[str, float]:
    """Uniform air/activity inputs for the PMV map and the handoff."""
    with st.expander("Air & occupant inputs (used for the PMV map and handoff)"):
        cols = st.columns(5)
        tdb = float(cols[0].slider("Air temperature [°C]", 10.0, 35.0, 24.0, 0.5, key="mrt_tdb"))
        rh = float(cols[1].slider("Relative humidity [%]", 0.0, 100.0, 50.0, 5.0, key="mrt_rh"))
        vr = float(cols[2].slider("Air speed [m/s]", 0.0, 1.0, 0.1, 0.05, key="mrt_vr"))
        met = float(cols[3].slider("Metabolic rate [met]", 0.8, 4.0, 1.2, 0.1, key="mrt_met"))
        clo = float(cols[4].slider("Clothing [clo]", 0.0, 2.0, 0.5, 0.1, key="mrt_clo"))
    return {"tdb": tdb, "rh": rh, "vr": vr, "met": met, "clo": clo}


def _pmv_over_grid(mrt: np.ndarray, air: dict[str, float]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = evaluate_thermal(
            tdb=np.full(mrt.size, air["tdb"]),
            tr=mrt.ravel(),
            vr=np.full(mrt.size, air["vr"]),
            rh=np.full(mrt.size, air["rh"]),
            met=air["met"],
            clo=air["clo"],
            category="none",
        )
    return res.pmv.reshape(mrt.shape)


def _solar_inputs(room: Room, x: float, y: float, posture: str) -> float:
    """Solar expander → ΔMRT at the occupant point (0 when disabled)."""
    with st.expander("Solar gain (short-wave)"):
        st.caption(
            "Applied at the occupant point only — direct-beam sun-patch "
            "geometry on the floor plan is out of scope."
        )
        if not room.windows:
            st.info("Activate a window above to include solar gain.")
            return 0.0
        if not st.checkbox("Apply solar ΔMRT at the occupant", key="mrt_solar_on"):
            return 0.0
        idx = int(
            st.selectbox(
                "Sun-lit window",
                range(len(room.windows)),
                format_func=lambda i: (
                    f"window {i} — {room.windows[i].wall} wall, "
                    f"{room.windows[i].width:g}×{room.windows[i].height:g} m"
                ),
                key="mrt_solar_win",
            )
        )
        win = room.windows[idx]
        c1, c2 = st.columns(2)
        sol_altitude = float(
            c1.slider("Solar altitude [°]", 0.0, 90.0, 45.0, 1.0, key="mrt_sol_alt")
        )
        sharp = float(
            c2.slider(
                "SHARP — sun angle relative to the person [°]",
                0.0,
                180.0,
                0.0,
                5.0,
                key="mrt_sharp",
                help="0 = sun in front of the person, 180 = behind.",
            )
        )
        sol_dir = float(
            c1.number_input(
                "Direct-beam solar radiation [W/m²]",
                0.0,
                1200.0,
                700.0,
                10.0,
                key="mrt_sol_dir",
            )
        )
        sol_trans = float(
            c2.slider("Solar transmittance", 0.0, 1.0, 0.7, 0.05, key="mrt_sol_trans")
        )
        f_bes = float(
            c1.slider(
                "Fraction of body exposed (f_bes)",
                0.0,
                1.0,
                0.5,
                0.05,
                key="mrt_f_bes",
            )
        )
        dist = {
            "front": x,
            "rear": room.length - x,
            "left": y,
            "right": room.width - y,
        }[win.wall]
        return solar_delta_mrt(
            win,
            dist,
            sol_altitude=sol_altitude,
            sharp=sharp,
            sol_radiation_dir=sol_dir,
            sol_transmittance=sol_trans,
            f_bes=f_bes,
            posture="sitting" if posture == "seated" else "standing",
        )


def render() -> None:
    st.title("MRT room designer")
    st.caption(
        "Plan-view mean radiant temperature of a rectangular room — "
        "Hottel view factors to six uniform-temperature surfaces, "
        "black-body surfaces. Up to two windows split off part of their "
        "parent wall at their own temperature."
    )

    length, width, height = _geometry_inputs()
    temps = _temps_inputs()
    windows = _window_inputs()
    try:
        room = Room(
            length=length,
            width=width,
            height=height,
            surface_temps=temps,
            windows=tuple(windows),
        )
    except ValueError as exc:
        st.error(f"Invalid geometry — {exc}")
        return

    posture = str(st.radio("Posture", ("seated", "standing"), horizontal=True, key="mrt_posture"))
    occ_z = 0.6 if posture == "seated" else 1.1
    c1, c2 = st.columns(2)
    occ_x = float(
        c1.slider(
            "Occupant x [m]",
            0.0,
            length,
            min(length / 2, length),
            0.05,
            key="mrt_ox",
        )
    )
    occ_y = float(
        c2.slider(
            "Occupant y [m]",
            0.0,
            width,
            min(width / 2, width),
            0.05,
            key="mrt_oy",
        )
    )

    air = _pmv_inputs()
    delta = _solar_inputs(room, occ_x, occ_y, posture)

    xs, ys, grid = _cached_grid(room, posture, _SPACING, _MARGIN)
    mode = st.radio("Map", ("MRT [°C]", "PMV"), horizontal=True, key="mrt_map_mode")
    if mode == "PMV":
        z_grid = _pmv_over_grid(grid, air)
        z_label, colorscale, zmid = "PMV", "RdBu_r", 0.0
    else:
        z_grid = grid
        z_label, colorscale, zmid = "MRT [°C]", "RdYlBu_r", None

    st.plotly_chart(
        _charts.mrt_plan_heatmap(
            xs,
            ys,
            z_grid,
            z_label=z_label,
            colorscale=colorscale,
            zmid=zmid,
            occupant_xy=(occ_x, occ_y),
            room_length=length,
            room_width=width,
            windows=[(w.wall, w.u0, w.width) for w in room.windows],
        ),
        width="stretch",
    )

    occ_mrt = mrt_point(room, occ_x, occ_y, occ_z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        occ_pmv = float(
            evaluate_thermal(
                tdb=np.array([air["tdb"]]),
                tr=np.array([occ_mrt]),
                vr=np.array([air["vr"]]),
                rh=np.array([air["rh"]]),
                met=air["met"],
                clo=air["clo"],
                category="none",
            ).pmv[0]
        )
    cols = st.columns(4)
    cols[0].metric("MRT at occupant", f"{occ_mrt:.1f} °C")
    cols[1].metric("Solar ΔMRT", f"+{delta:.1f} °C" if delta else "0 °C")
    cols[2].metric(
        "PMV at occupant",
        "outside model range" if np.isnan(occ_pmv) else f"{occ_pmv:+.2f}",
    )
    cols[3].metric(
        "Effective MRT",
        f"{occ_mrt + delta:.1f} °C" if delta else f"{occ_mrt:.1f} °C",
    )

    if _state.tool_enabled("calculator") and st.button("Use in calculator", key="mrt_handoff"):
        _state.set(
            _state.CALC_INPUTS,
            {
                "tdb": air["tdb"],
                "tr": occ_mrt + delta,
                "rh": air["rh"],
                "vr": air["vr"],
                "met": air["met"],
                "clo": air["clo"],
            },
        )
        _state.goto("calculator")

    st.caption(
        "View factors: Hottel (1954) port of the MIT-licensed "
        "mrt_replication project; PMV via pythermalcomfort "
        "(Tartarini & Schiavon 2020, doi:10.1016/j.softx.2020.100578); "
        "solar gain per ASHRAE 55-2023 §C2. Surfaces are treated as "
        "black bodies at uniform temperature; MRT ignores air "
        "temperature — set 'Air temperature' in the expander for PMV."
    )
