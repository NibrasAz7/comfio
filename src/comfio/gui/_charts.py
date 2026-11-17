"""Plotly chart builders for the comfio Streamlit app."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DOMAIN_COLORS = {
    "thermal": "#e76f51",
    "visual": "#e9c46a",
    "acoustic": "#2a9d8f",
    "iaq": "#457b9d",
}


def gauge(value: float, title: str = "Global IEQ Index") -> go.Figure:
    """Big verdict gauge (0–100) with comfort-band color zones."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#264653"},
                "steps": [
                    {"range": [0, 50], "color": "#f4a9a0"},
                    {"range": [50, 80], "color": "#f6e8a6"},
                    {"range": [80, 100], "color": "#a8e0b3"},
                ],
                "threshold": {
                    "line": {"color": "#264653", "width": 4},
                    "value": 80,
                },
            },
        )
    )
    fig.update_layout(height=240, margin={"l": 30, "r": 30, "t": 50, "b": 10})
    return fig


def domain_bars(scores: dict[str, np.ndarray], threshold: float = 80.0) -> go.Figure:
    """Per-domain mean score bars with the compliance threshold line."""
    names = list(scores)
    means = [float(np.nanmean(scores[d])) for d in names]
    colors = [DOMAIN_COLORS.get(d, "#888") for d in names]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=means,
            marker_color=colors,
            text=[f"{m:.0f}" for m in means],
            textposition="auto",
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#c0392b")
    fig.update_layout(
        height=260,
        yaxis_range=[0, 100],
        margin={"l": 40, "r": 20, "t": 20, "b": 30},
        showlegend=False,
    )
    return fig


def comfort_scatter(
    tdb: np.ndarray,
    rh: np.ndarray,
    pmv: np.ndarray | None = None,
) -> go.Figure:
    """Temperature–humidity scatter colored by PMV.

    Overlays the ASHRAE-55 standard comfort zone (operative temp
    20–27.5 °C, RH 20–60 % at ~0.5–1.0 clo, still air) as a polygon
    computed from the standard's limits.
    """
    fig = go.Figure()
    # ASHRAE 55 standard zone polygon (sedentary, 0.5-1.0 clo, v<0.2 m/s).
    zx = [20, 27.5, 27.5, 20]
    zy = [20, 20, 60, 60]
    fig.add_trace(
        go.Scatter(
            x=zx,
            y=zy,
            fill="toself",
            fillcolor="rgba(120, 200, 140, 0.18)",
            line={"color": "rgba(60, 160, 90, 0.6)", "dash": "dash"},
            name="ASHRAE 55 comfort zone",
            mode="lines",
        )
    )
    marker = (
        {
            "color": pmv,
            "colorscale": "RdBu_r",
            "cmin": -1.5,
            "cmax": 1.5,
            "size": 6,
            "colorbar": {"title": "PMV"},
        }
        if pmv is not None
        else {"color": "#264653", "size": 6}
    )
    fig.add_trace(go.Scatter(x=tdb, y=rh, mode="markers", marker=marker, name="samples"))
    fig.update_layout(
        height=380,
        xaxis_title="Air / operative temperature [°C]",
        yaxis_title="Relative humidity [%]",
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def timeseries(
    x: pd.Series | np.ndarray,
    y: np.ndarray,
    name: str,
    color: str,
    threshold: float | None = None,
) -> go.Figure:
    """Single-metric time series with an optional threshold line."""
    fig = go.Figure(
        go.Scatter(x=x, y=y, mode="lines", name=name, line={"color": color, "width": 1})
    )
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="#c0392b")
    fig.update_layout(height=240, margin={"l": 50, "r": 20, "t": 20, "b": 40})
    return fig


def sweep_curve(rows: list[dict], x_label: str) -> go.Figure:
    """What-if result: compliance rate + IEQ mean vs a swept parameter."""
    x = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[r["compliance_pct"] for r in rows],
            mode="lines+markers",
            name="Compliance rate %",
            line={"color": "#2a9d8f", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[r["ieq_mean"] for r in rows],
            mode="lines+markers",
            name="IEQ mean",
            line={"color": "#264653", "width": 2, "dash": "dash"},
        )
    )
    if any(not np.isnan(r.get("domain_mean", np.nan)) for r in rows):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[r["domain_mean"] for r in rows],
                mode="lines+markers",
                name="Thermal pass %",
                line={"color": "#e76f51", "width": 2, "dash": "dot"},
            )
        )
    fig.update_layout(
        height=320,
        xaxis_title=x_label,
        yaxis_range=[0, 105],
        yaxis_title="Score / rate (0–100)",
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def pmv_tsv_scatter(pmv: np.ndarray, tsv: np.ndarray, alpha: float, beta: float) -> go.Figure:
    """PMV-vs-vote scatter with the OLS personalisation fit line."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pmv,
            y=tsv,
            mode="markers",
            marker={"color": "#e76f51", "size": 6, "opacity": 0.7},
            name="samples",
        )
    )
    x_line = np.linspace(float(np.nanmin(pmv)), float(np.nanmax(pmv)), 50)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=alpha * x_line + beta,
            mode="lines",
            line={"color": "#264653", "width": 2},
            name=f"TSV = {alpha:.2f}·PMV + {beta:.2f}",
        )
    )
    fig.update_layout(
        height=320,
        xaxis_title="Model PMV",
        yaxis_title="Occupant vote (TSV)",
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def scenario_bars(scenarios: list[dict]) -> go.Figure:
    """Grouped bars comparing pinned scenarios: IEQ mean + domain means."""
    names = [s["name"] for s in scenarios]
    domains = sorted({d for s in scenarios for d in s["domain_means"]})
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="IEQ index",
            x=names,
            y=[s["ieq_mean"] for s in scenarios],
            marker_color="#264653",
        )
    )
    for d in domains:
        fig.add_trace(
            go.Bar(
                name=d,
                x=names,
                y=[s["domain_means"].get(d) for s in scenarios],
                marker_color=DOMAIN_COLORS.get(d, "#888"),
            )
        )
    fig.update_layout(
        barmode="group",
        height=320,
        yaxis_range=[0, 100],
        yaxis_title="Mean score",
        margin={"l": 50, "r": 20, "t": 30, "b": 40},
    )
    return fig


def exceedance_heatmap(
    timestamps: pd.Series,
    index: np.ndarray,
    threshold: float = 80.0,
) -> go.Figure:
    """Hour-of-day × day exceedance heatmap ('when does it fail')."""
    hours = pd.Series(timestamps).dt.hour.to_numpy()
    dates = pd.Series(timestamps).dt.date.astype(str).to_numpy()
    fail = (np.asarray(index, dtype=float) < threshold).astype(float)
    uniq_h = np.arange(24)
    uniq_d = np.unique(dates)
    grid = np.full((len(uniq_d), 24), np.nan)
    hpos = {int(h): i for i, h in enumerate(uniq_h)}
    dpos = {str(d): i for i, d in enumerate(uniq_d)}
    for h, d, f in zip(hours, dates, fail, strict=True):
        grid[dpos[str(d)], hpos[int(h)]] = f
    fig = go.Figure(
        go.Heatmap(
            z=grid,
            x=[f"{h:02d}:00" for h in uniq_h],
            y=uniq_d,
            colorscale=[[0, "#a8e0b3"], [1, "#e76f51"]],
            showscale=False,
        )
    )
    fig.update_layout(
        height=max(200, min(500, 24 * len(uniq_d))),
        xaxis_title="Hour of day",
        margin={"l": 90, "r": 20, "t": 20, "b": 40},
    )
    return fig


def psychro_chart(
    zones: Sequence[tuple[str, pd.DataFrame]],
    points: Sequence[tuple[str, float, float]],
) -> go.Figure:
    """Psychrometric chart: RH lines + PMV comfort zones + points.

    X is operative temperature [°C], Y the humidity ratio [g/kg]. Each
    zone is a ``pmv_zone_boundary`` frame (columns ``w``, ``t_cool``,
    ``t_warm``) drawn as a closed polygon; points are
    ``(label, operative_c, w_kg_per_kg)``.
    """
    from comfio.gui.tools.calculator._psychro import humidity_ratio

    t_axis = np.linspace(10.0, 40.0, 61)
    fig = go.Figure()
    for rh_pct in range(10, 100, 10):
        w_line = humidity_ratio(t_axis, np.full_like(t_axis, float(rh_pct))) * 1000.0
        fig.add_trace(
            go.Scatter(
                x=t_axis,
                y=w_line,
                mode="lines",
                line={"color": "#d0d0d0", "width": 1},
                name=f"RH {rh_pct} %",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    w_sat = humidity_ratio(t_axis, np.full_like(t_axis, 100.0)) * 1000.0
    fig.add_trace(
        go.Scatter(
            x=t_axis,
            y=w_sat,
            mode="lines",
            line={"color": "#888888", "width": 1.5},
            name="Saturation (RH 100 %)",
        )
    )
    zone_lines = ["#2a9d8f", "#e76f51", "#457b9d"]
    zone_fills = [
        "rgba(42,157,143,0.2)",
        "rgba(231,111,81,0.2)",
        "rgba(69,123,157,0.2)",
    ]
    for i, (label, zone) in enumerate(zones):
        z = zone.dropna(subset=["t_cool", "t_warm"]).sort_values("w")
        if z.empty:
            continue
        xs = list(z["t_cool"]) + list(z["t_warm"].iloc[::-1])
        ys = list(z["w"] * 1000.0) + list((z["w"] * 1000.0).iloc[::-1])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=zone_fills[i % len(zone_fills)],
                line={"color": zone_lines[i % len(zone_lines)], "width": 1.5},
                name=label,
            )
        )
    for label, t_op, w in points:
        fig.add_trace(
            go.Scatter(
                x=[t_op],
                y=[w * 1000.0],
                mode="markers",
                marker={
                    "symbol": "star",
                    "size": 14,
                    "color": "#ffd700",
                    "line": {"color": "#264653", "width": 1.5},
                },
                name=label,
            )
        )
    fig.update_layout(
        height=420,
        xaxis_title="Operative temperature [°C]",
        xaxis_range=[10, 40],
        yaxis_title="Humidity ratio [g/kg]",
        yaxis_range=[0, 30],
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def adaptive_chart(
    method: Literal["ashrae", "en"],
    point: tuple[float, float] | None,
) -> go.Figure:
    """Adaptive-model comfort bands over the outdoor-temperature axis.

    ASHRAE 55: prevailing mean outdoor temperature, 80 % and 90 % bands.
    EN 16798-1: running mean outdoor temperature, categories I–III.
    ``point`` is ``(t_outdoor_metric, operative_c)`` or None.
    """
    from comfio.domains.thermal_adaptive import (
        ASHRAE_T_OUT_MAX,
        ASHRAE_T_OUT_MIN,
        EN_T_OUT_MAX,
        EN_T_OUT_MIN,
        evaluate_adaptive_ashrae,
        evaluate_adaptive_en,
    )

    fig = go.Figure()
    t_dummy = np.array([22.0])
    if method == "ashrae":
        grid = np.linspace(ASHRAE_T_OUT_MIN, ASHRAE_T_OUT_MAX, 24)
        bands = {}
        for acc in (80, 90):
            lows, highs = [], []
            for t in grid:
                r = evaluate_adaptive_ashrae(
                    tdb=t_dummy, tr=t_dummy, t_prevail=float(t), acceptability=acc
                )
                lows.append(r.t_comf_lower)
                highs.append(r.t_comf_upper)
            bands[f"{acc} %"] = (lows, highs)
        t_comf = 0.31 * grid + 17.8
        x_title = "Prevailing mean outdoor temperature [°C]"
    else:
        grid = np.linspace(EN_T_OUT_MIN, EN_T_OUT_MAX, 21)
        bands = {}
        for cat in ("i", "ii", "iii"):
            lows, highs = [], []
            for t in grid:
                r = evaluate_adaptive_en(
                    tdb=t_dummy, tr=t_dummy, t_running_mean=float(t), category=cat
                )
                lows.append(r.t_comf_lower)
                highs.append(r.t_comf_upper)
            bands[f"Cat {cat.upper()}"] = (lows, highs)
        t_comf = 0.33 * grid + 18.8
        x_title = "Running mean outdoor temperature [°C]"

    fills = [
        "rgba(42,157,143,0.15)",
        "rgba(233,196,106,0.15)",
        "rgba(231,111,81,0.15)",
    ]
    lines = ["#2a9d8f", "#e9c46a", "#e76f51"]
    for i, (label, (lows, highs)) in enumerate(bands.items()):
        xs = list(grid) + list(grid[::-1])
        ys = list(highs) + list(lows[::-1])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                fillcolor=fills[i % len(fills)],
                line={"color": lines[i % len(lines)], "width": 1},
                name=label,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(grid),
            y=list(t_comf),
            mode="lines",
            line={"color": "#264653", "width": 2},
            name="Comfort temperature",
        )
    )
    if point is not None:
        fig.add_trace(
            go.Scatter(
                x=[point[0]],
                y=[point[1]],
                mode="markers",
                marker={
                    "symbol": "star",
                    "size": 14,
                    "color": "#ffd700",
                    "line": {"color": "#264653", "width": 1.5},
                },
                name="you",
            )
        )
    fig.update_layout(
        height=380,
        xaxis_title=x_title,
        yaxis_title="Operative temperature [°C]",
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def db_adaptive_scatter(
    df: pd.DataFrame,
    overlay: pd.DataFrame | None = None,
    max_points: int = 20000,
) -> go.Figure:
    """DB II operative-vs-outdoor scatter with the ASHRAE adaptive band.

    X is ``t_mot_isd`` (the database's 7-day running mean outdoor
    temperature), Y the ``top`` operative temperature, colour the thermal
    sensation vote. ``overlay`` (a ``user_overlay`` frame with ``top`` +
    ``t_mot``) is drawn as black stars. Large frames are deterministically
    subsampled to ``max_points`` for rendering.
    """
    from comfio.domains.thermal_adaptive import (
        ASHRAE_T_OUT_MAX,
        ASHRAE_T_OUT_MIN,
        Acceptability,
        evaluate_adaptive_ashrae,
    )

    fig = go.Figure()
    d = df.dropna(subset=["t_mot_isd", "top"])
    if len(d) > max_points:
        d = d.sample(n=max_points, random_state=0)
    if "thermal_sensation" in d.columns:
        marker: dict = {
            "color": d["thermal_sensation"],
            "colorscale": "RdBu_r",
            "cmin": -3,
            "cmax": 3,
            "size": 4,
            "opacity": 0.5,
            "colorbar": {"title": "TSV"},
        }
    else:
        marker = {"color": "#264653", "size": 4, "opacity": 0.5}
    fig.add_trace(
        go.Scatter(
            x=d["t_mot_isd"],
            y=d["top"],
            mode="markers",
            marker=marker,
            name="DB II votes",
        )
    )

    grid = np.linspace(ASHRAE_T_OUT_MIN, ASHRAE_T_OUT_MAX, 24)
    t_dummy = np.array([22.0])
    band_styles: tuple[tuple[Acceptability, str, str], ...] = (
        (80, "rgba(42,157,143,0.12)", "#2a9d8f"),
        (90, "rgba(233,196,106,0.12)", "#b8912a"),
    )
    for acc, fill, line in band_styles:
        lows, highs = [], []
        for t in grid:
            r = evaluate_adaptive_ashrae(
                tdb=t_dummy, tr=t_dummy, t_prevail=float(t), acceptability=acc
            )
            lows.append(r.t_comf_lower)
            highs.append(r.t_comf_upper)
        fig.add_trace(
            go.Scatter(
                x=list(grid) + list(grid[::-1]),
                y=list(highs) + list(lows[::-1]),
                fill="toself",
                fillcolor=fill,
                line={"color": line, "width": 1},
                name=f"ASHRAE {acc} % band",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(grid),
            y=list(0.31 * grid + 17.8),
            mode="lines",
            line={"color": "#264653", "width": 1.5, "dash": "dash"},
            name="ASHRAE comfort temp",
        )
    )

    if overlay is not None and {"top", "t_mot"} <= set(overlay.columns):
        ov = overlay.dropna(subset=["top", "t_mot"])
        if not ov.empty:
            fig.add_trace(
                go.Scatter(
                    x=ov["t_mot"],
                    y=ov["top"],
                    mode="markers",
                    marker={
                        "symbol": "star",
                        "size": 9,
                        "color": "#111111",
                    },
                    name="your data",
                )
            )
    fig.update_layout(
        height=420,
        xaxis_title="Outdoor temperature — 7-day running mean (t_mot_isd) [°C]",
        yaxis_title="Operative temperature [°C]",
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def sensation_bins_chart(bins: pd.DataFrame) -> go.Figure:
    """Mean TSV (line) and % acceptable (bars) per operative-temp bin."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=bins["t_bin"],
            y=bins["pct_acceptable"],
            name="% acceptable",
            marker_color="#a8e0b3",
            opacity=0.65,
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=bins["t_bin"],
            y=bins["mean_tsv"],
            mode="lines+markers",
            name="mean TSV",
            line={"color": "#264653", "width": 2},
        ),
        secondary_y=False,
    )
    fig.update_yaxes(title_text="Mean TSV [-3..+3]", secondary_y=False)
    fig.update_yaxes(title_text="% acceptable", secondary_y=True, range=[0, 100])
    fig.update_xaxes(title_text="Operative temperature bin [°C]")
    fig.update_layout(
        height=360,
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def pmv_bias_chart(bias: pd.DataFrame) -> go.Figure:
    """Mean (PMV − TSV) per cooling type — model-vs-vote bias."""
    fig = go.Figure(
        go.Bar(
            x=bias["cooling_type"],
            y=bias["mean_bias"],
            marker_color="#457b9d",
            text=[f"n={n}" for n in bias["n"]],
            textposition="auto",
        )
    )
    fig.add_hline(y=0, line_color="#264653", line_width=1)
    fig.update_layout(
        height=320,
        xaxis_title="Cooling type",
        yaxis_title="Mean (PMV − TSV)",
        margin={"l": 50, "r": 20, "t": 30, "b": 40},
    )
    return fig


def ranges_curve(df: pd.DataFrame, param: str, label: str) -> go.Figure:
    """PMV / SET vs a swept input, PPD on a secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df[param],
            y=df["pmv"],
            mode="lines+markers",
            name="PMV",
            line={"color": "#264653", "width": 2},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df[param],
            y=df["set_c"],
            mode="lines+markers",
            name="SET [°C]",
            line={"color": "#e76f51", "width": 2, "dash": "dash"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df[param],
            y=df["ppd"],
            mode="lines+markers",
            name="PPD [%]",
            line={"color": "#2a9d8f", "width": 1.5, "dash": "dot"},
        ),
        secondary_y=True,
    )
    fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="#a8e0b3", opacity=0.25, line_width=0)
    fig.update_yaxes(title_text="PMV / SET [°C]", secondary_y=False)
    fig.update_yaxes(title_text="PPD [%]", secondary_y=True, range=[0, 100])
    fig.update_xaxes(title_text=label)
    fig.update_layout(
        height=380,
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def mrt_plan_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    grid: np.ndarray,
    *,
    z_label: str,
    colorscale: str = "RdYlBu_r",
    zmid: float | None = None,
    occupant_xy: tuple[float, float] | None = None,
    room_length: float | None = None,
    room_width: float | None = None,
    windows: Sequence[tuple[str, float, float]] = (),
) -> go.Figure:
    """Plan-view heatmap of an ``mrt_grid`` result.

    ``xs``/``ys``/``grid`` come from ``thermal_mrt.mrt_grid`` (grid is
    ``[iy, ix]``). ``windows`` are ``(wall, u0, width)`` triples drawn as
    thick segments on the matching wall; ``occupant_xy`` is a star
    marker. When room extents are given the wall outline is drawn and
    the axes keep a 1:1 aspect ratio.
    """
    fig = go.Figure(
        go.Heatmap(
            x=xs,
            y=ys,
            z=grid,
            colorscale=colorscale,
            zmid=zmid,
            colorbar={"title": z_label},
        )
    )
    if room_length is not None and room_width is not None:
        length, width = room_length, room_width
        fig.add_trace(
            go.Scatter(
                x=[0, length, length, 0, 0],
                y=[0, 0, width, width, 0],
                mode="lines",
                line={"color": "#264653", "width": 2},
                name="walls",
                hoverinfo="skip",
            )
        )
        for wall, u0, w in windows:
            seg_x: list[float]
            seg_y: list[float]
            if wall == "front":
                seg_x, seg_y = [0.0, 0.0], [u0, u0 + w]
            elif wall == "rear":
                seg_x, seg_y = [length, length], [u0, u0 + w]
            elif wall == "left":
                seg_x, seg_y = [u0, u0 + w], [0.0, 0.0]
            else:  # right
                seg_x, seg_y = [u0, u0 + w], [width, width]
            fig.add_trace(
                go.Scatter(
                    x=seg_x,
                    y=seg_y,
                    mode="lines",
                    line={"color": "#457b9d", "width": 7},
                    name=f"window ({wall})",
                    hoverinfo="name",
                )
            )
    if occupant_xy is not None:
        fig.add_trace(
            go.Scatter(
                x=[occupant_xy[0]],
                y=[occupant_xy[1]],
                mode="markers",
                marker={
                    "symbol": "star",
                    "size": 15,
                    "color": "#ffd700",
                    "line": {"color": "#264653", "width": 1.5},
                },
                name="occupant",
            )
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=480,
        xaxis_title="x — room length [m]",
        yaxis_title="y — room width [m]",
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )
    return fig


def satisfaction_bars(summary: pd.DataFrame) -> go.Figure:
    """Diverging stacked bars of a survey ``summarize`` result.

    One row per satisfaction category: % dissatisfied extends left,
    neutral is centred on zero, % satisfied extends right. When the
    summary carries more than one group, each group gets its own
    subplot row (facet).
    """
    from comfio.gui.tools.survey._logic import SAT_COLUMNS

    groups = list(dict.fromkeys(summary["group"].tolist())) or ["all"]
    facet = len(groups) > 1
    fig = make_subplots(
        rows=len(groups),
        cols=1,
        shared_xaxes=True,
        subplot_titles=groups if facet else None,
        vertical_spacing=min(0.08, 0.5 / max(len(groups) - 1, 1)),
    )
    for row_idx, group in enumerate(groups, start=1):
        sub = summary[summary["group"] == group]
        labels = [SAT_COLUMNS.get(str(c), str(c)) for c in sub["category"]]
        dis = -sub["pct_dissatisfied"].to_numpy(dtype=float)
        neu = sub["pct_neutral"].to_numpy(dtype=float)
        sat = sub["pct_satisfied"].to_numpy(dtype=float)
        first = row_idx == 1
        fig.add_trace(
            go.Bar(
                y=labels,
                x=dis,
                base=-neu / 2,
                orientation="h",
                name="dissatisfied",
                legendgroup="dis",
                showlegend=first,
                marker_color="#e76f51",
                text=[f"{-v:.0f}" if v < 4 else "" for v in dis],
                textposition="inside",
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                y=labels,
                x=neu,
                base=-neu / 2,
                orientation="h",
                name="neutral",
                legendgroup="neu",
                showlegend=first,
                marker_color="#e9c46a",
                text=[f"{v:.0f}" if v >= 8 else "" for v in neu],
                textposition="inside",
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                y=labels,
                x=sat,
                base=neu / 2,
                orientation="h",
                name="satisfied",
                legendgroup="sat",
                showlegend=first,
                marker_color="#a8e0b3",
                text=[f"{v:.0f}" if v >= 4 else "" for v in sat],
                textposition="inside",
            ),
            row=row_idx,
            col=1,
        )
    ticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
    fig.update_xaxes(
        range=[-100, 100],
        tickvals=ticks,
        ticktext=[str(abs(t)) for t in ticks],
        zeroline=True,
        zerolinecolor="#264653",
        zerolinewidth=1,
    )
    fig.update_layout(
        barmode="overlay",
        height=max(260, min(2400, 90 * len(summary) + 140)),
        xaxis_title="% of respondents",
        margin={"l": 50, "r": 20, "t": 40, "b": 50},
    )
    return fig
