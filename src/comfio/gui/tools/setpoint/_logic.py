"""Setpoint explorer logic — comfort-band edges and the deadband table.

Band edges are the operative temperatures (``tdb == tr``) where the
ASHRAE 55-2023 PMV — the calculator's :func:`ashrae_pmv` with dynamic
clo and relative air speed — crosses ±``pmv_limit``, found with
``scipy.optimize.brentq`` over 10–36 °C. A missing crossing returns NaN;
edges are never clamped or extrapolated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from comfio.gui.tools.calculator._models import ashrae_pmv

ENERGY_CALC_URL = "https://energy-calc-2wmjqjatpa-uc.a.run.app/"
HOYT_2015 = (
    "Hoyt, Arens & Zhang (2015), Building and Environment 88:89–96, "
    "doi:10.1016/j.buildenv.2014.09.010"
)

_T_LO, _T_HI = 10.0, 36.0


def _pmv_root(
    met: float,
    clo: float,
    vr: float,
    rh: float,
    target: float,
) -> float:
    """Operative temperature where ASHRAE PMV = ``target``; NaN if none."""

    def f(t: float) -> float:
        return (
            float(
                ashrae_pmv(
                    np.array([t]),
                    np.array([t]),
                    np.array([vr]),
                    np.array([rh]),
                    np.array([met]),
                    np.array([clo]),
                )[0]
            )
            - target
        )

    try:
        return float(brentq(f, _T_LO, _T_HI))
    except (ValueError, RuntimeError):
        return np.nan


def comfort_band(
    met: float,
    clo_heating: float,
    clo_cooling: float,
    rh: float,
    vr_still: float = 0.1,
    vr_fan: float | None = None,
    pmv_limit: float = 0.5,
) -> dict[str, float]:
    """Comfort-band edges on the operative-temperature axis [°C].

    ``heat_min_c`` — PMV crossing −``pmv_limit`` at ``clo_heating`` /
    ``vr_still``; ``cool_max_c`` — PMV crossing +``pmv_limit`` at
    ``clo_cooling`` / ``vr_still``; ``cool_max_fan_c`` — the same with
    ``vr_fan`` (NaN when ``vr_fan`` is None or no root exists).
    """
    return {
        "heat_min_c": _pmv_root(met, clo_heating, vr_still, rh, -pmv_limit),
        "cool_max_c": _pmv_root(met, clo_cooling, vr_still, rh, pmv_limit),
        "cool_max_fan_c": (
            _pmv_root(met, clo_cooling, vr_fan, rh, pmv_limit)
            if vr_fan is not None
            else np.nan
        ),
    }


def deadband_table(
    current_heat: float,
    current_cool: float,
    band: dict[str, float],
) -> pd.DataFrame:
    """Current vs proposed setpoints plus a delta row (the widening).

    The proposed cooling setpoint is ``cool_max_fan_c`` when a fan edge
    exists, else ``cool_max_c``. Delta values are proposed − current:
    a negative heating delta means the heating setpoint drops.
    """
    proposed_cool = band["cool_max_fan_c"]
    fan_used = not np.isnan(proposed_cool)
    if not fan_used:
        proposed_cool = band["cool_max_c"]
    proposed_heat = band["heat_min_c"]
    current_width = current_cool - current_heat
    proposed_width = proposed_cool - proposed_heat
    return pd.DataFrame(
        [
            {
                "scenario": "current",
                "heating setpoint": current_heat,
                "cooling setpoint": current_cool,
                "deadband width": current_width,
                "note": "as configured",
            },
            {
                "scenario": "proposed",
                "heating setpoint": proposed_heat,
                "cooling setpoint": proposed_cool,
                "deadband width": proposed_width,
                "note": (
                    "PMV ±0.5 band edges"
                    + (", fan cooling edge" if fan_used else "")
                ),
            },
            {
                "scenario": "delta",
                "heating setpoint": proposed_heat - current_heat,
                "cooling setpoint": proposed_cool - current_cool,
                "deadband width": proposed_width - current_width,
                "note": "proposed − current; positive width = wider deadband",
            },
        ]
    )
