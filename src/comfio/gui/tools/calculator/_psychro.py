"""Psychrometrics + PMV comfort-zone boundary for the calculator tool.

The zone boundary is found on a vectorised two-stage grid: a coarse
1 °C scan brackets each PMV crossing, then a fine 0.1 °C rescan inside
the bracket gives the interpolated boundary. One batched
``ashrae_pmv`` call per stage keeps chart updates interactive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from comfio.gui.tools.calculator._models import ashrae_pmv

_T_LO, _T_HI, _T_STEP = 10.0, 40.0, 0.1
_RATIO = 0.621945  # Mw/Ma
_P_ATM = 101325.0


def humidity_ratio(tdb: np.ndarray, rh: np.ndarray, p_atm: float = _P_ATM) -> np.ndarray:
    """Humidity ratio [kg/kg] from dry-bulb temperature and RH."""
    from pythermalcomfort.utilities import p_sat

    pv = (
        np.asarray(rh, dtype=float)
        / 100.0
        * np.asarray(p_sat(np.asarray(tdb, dtype=float)), dtype=float)
    )
    return np.asarray(_RATIO * pv / (p_atm - pv), dtype=float)


def _rh_from_w(w: np.ndarray | float, t: np.ndarray | float) -> np.ndarray:
    """RH [%] implied by humidity ratio w at temperature t (p = 101325 Pa)."""
    from pythermalcomfort.utilities import p_sat

    wv = np.asarray(w, dtype=float)
    pv = wv * _P_ATM / (_RATIO + wv)
    return np.asarray(100.0 * pv / np.asarray(p_sat(t), dtype=float))


def _crossing(t: np.ndarray, f: np.ndarray) -> float:
    """First linear-interpolated zero of f along t; NaN if none."""
    for j in range(len(t) - 1):
        if not (np.isfinite(f[j]) and np.isfinite(f[j + 1])):
            continue
        if f[j] == 0.0:
            return float(t[j])
        if f[j] * f[j + 1] < 0:
            return float(t[j] - f[j] * (t[j + 1] - t[j]) / (f[j + 1] - f[j]))
    return float("nan")


def _bracket(f: np.ndarray) -> int | None:
    """Index of the first sign change (or exact zero) in f, else None."""
    for j in range(len(f) - 1):
        if not (np.isfinite(f[j]) and np.isfinite(f[j + 1])):
            continue
        if f[j] == 0.0 or f[j] * f[j + 1] < 0:
            return j
    return None


def pmv_zone_boundary(
    met: float,
    clo: float,
    vr: float,
    *,
    pmv_limit: float = 0.5,
    w_max: float = 0.03,
    n: int = 40,
    dynamic_clo: bool = True,
    relative_air_speed: bool = True,
) -> pd.DataFrame:
    """PMV comfort-zone boundary on the psychrometric chart.

    For each humidity ratio ``w`` in ``linspace(0, w_max, n)`` the
    operative temperature where ASHRAE PMV crosses −``pmv_limit``
    (``t_cool``) and +``pmv_limit`` (``t_warm``), with NaN where no
    crossing exists. ``tdb == tr`` (operative-temperature axis).
    """
    w = np.linspace(0.0, w_max, n)
    t_coarse = np.arange(_T_LO, _T_HI + 0.5, 1.0)

    # Stage 1 — coarse scan over the whole (w × t) grid.
    wg, tg = np.meshgrid(w, t_coarse, indexing="ij")
    rh = _rh_from_w(wg, tg)
    pmv = np.where(
        rh <= 100.0,
        ashrae_pmv(
            tg,
            tg,
            vr,
            rh,
            met,
            clo,
            dynamic_clo=dynamic_clo,
            relative_air_speed=relative_air_speed,
        ),
        np.nan,
    )

    # Stage 2 — one batched fine scan inside every bracketed interval.
    targets = {"t_cool": -pmv_limit, "t_warm": pmv_limit}
    jobs: list[tuple[int, str, int]] = []  # (w-row, column, coarse index)
    for i in range(n):
        for key, target in targets.items():
            j = _bracket(pmv[i] - target)
            if j is not None:
                jobs.append((i, key, j))

    fine_t: list[float] = []
    fine_w: list[float] = []
    for i, _, j in jobs:
        span = t_coarse[j] + np.arange(0.0, 1.0 + _T_STEP / 2, _T_STEP)
        fine_t.extend(span.tolist())
        fine_w.extend([w[i]] * len(span))
    if fine_t:
        tf = np.asarray(fine_t)
        wf = np.asarray(fine_w)
        rh_f = _rh_from_w(wf, tf)
        fine_pmv = np.where(
            rh_f <= 100.0,
            ashrae_pmv(
                tf,
                tf,
                vr,
                rh_f,
                met,
                clo,
                dynamic_clo=dynamic_clo,
                relative_air_speed=relative_air_speed,
            ),
            np.nan,
        )
    else:
        fine_pmv = np.empty(0)

    rows = [{"w": float(wi), "t_cool": np.nan, "t_warm": np.nan} for wi in w]
    k = 0
    span_n = len(np.arange(0.0, 1.0 + _T_STEP / 2, _T_STEP))
    for i, key, j in jobs:
        tf = t_coarse[j] + np.arange(0.0, 1.0 + _T_STEP / 2, _T_STEP)
        f = fine_pmv[k : k + span_n] - targets[key]
        rows[i][key] = _crossing(tf, f)
        k += span_n
    return pd.DataFrame(rows)
