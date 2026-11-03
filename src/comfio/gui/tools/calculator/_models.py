"""Comfort-model wrappers for the calculator tool.

All evaluation goes through ``pythermalcomfort`` (ASHRAE 55-2023 PMV,
SET, cooling effect) or comfio's own domain functions (ISO 7730 PMV,
ASHRAE/EN adaptive). Inputs outside a model's applicability domain
produce ``NaN`` plus a note — never a clamped or extrapolated value.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from comfio.domains.thermal import CATEGORY_PMV_LIMITS, evaluate_thermal
from comfio.domains.thermal_adaptive import (
    ASHRAE_T_OUT_MAX,
    ASHRAE_T_OUT_MIN,
    EN_T_OUT_MAX,
    EN_T_OUT_MIN,
    ENCategory,
    evaluate_adaptive_ashrae,
    evaluate_adaptive_en,
)

METHODS = (
    "ASHRAE 55-2023 PMV",
    "ISO 7730-2005 PMV",
    "EN 16798-1 PMV",
    "ASHRAE 55 adaptive",
    "EN 16798-1 adaptive",
)
PMV_METHODS = METHODS[:3]
ADAPTIVE_METHODS = METHODS[3:]

# EN 16798-1:2019 PMV categories I-III (category IV pending user confirmation)
EN16798_PMV_LIMITS: dict[str, float] = {"I": 0.2, "II": 0.5, "III": 0.7}


@dataclass(frozen=True)
class Inputs:
    tdb: float  # dry-bulb air temperature [°C]
    tr: float  # mean radiant temperature [°C]
    rh: float  # relative humidity [%]
    vr: float  # air speed [m/s]
    met: float  # metabolic rate [met]
    clo: float  # clothing insulation [clo]
    t_running_mean: float | None = None  # outdoor metric for adaptive models [°C]


@dataclass(frozen=True)
class PointResult:
    method: str
    pmv: float
    ppd: float
    set_c: float  # Standard Effective Temperature [°C] (ASHRAE 55)
    ce: float  # cooling effect [°C] (ASHRAE 55)
    compliant: bool | None
    category: str | None
    notes: tuple[str, ...]
    extra: Mapping[str, float] = field(default_factory=dict)


def _nan_result(method: str, note: str) -> PointResult:
    return PointResult(
        method=method,
        pmv=np.nan,
        ppd=np.nan,
        set_c=np.nan,
        ce=np.nan,
        compliant=None,
        category=None,
        notes=(note,),
    )


def _preprocessed(
    vr: float | np.ndarray,
    met: float | np.ndarray,
    clo: float | np.ndarray,
    dynamic_clo: bool,
    relative_air_speed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """ASHRAE 55-2023 preprocessing: relative air speed + dynamic clo."""
    from pythermalcomfort.utilities import clo_dynamic_ashrae, v_relative

    vr_rel = (
        np.asarray(v_relative(vr, met), dtype=float)
        if relative_air_speed
        else np.asarray(vr, dtype=float)
    )
    clo_d = (
        np.asarray(clo_dynamic_ashrae(clo, met, model="55-2023"), dtype=float)
        if dynamic_clo
        else np.asarray(clo, dtype=float)
    )
    return vr_rel, clo_d


def ashrae_pmv(
    tdb: np.ndarray | float,
    tr: np.ndarray | float,
    vr: np.ndarray | float,
    rh: np.ndarray | float,
    met: np.ndarray | float,
    clo: np.ndarray | float,
    *,
    dynamic_clo: bool = True,
    relative_air_speed: bool = True,
) -> np.ndarray:
    """Vectorised ASHRAE 55-2023 PMV (dynamic clo + relative air speed)."""
    from pythermalcomfort.models import pmv_ppd_ashrae

    tdb_a, tr_a, vr_a, rh_a, met_a, clo_a = np.broadcast_arrays(
        np.asarray(tdb, dtype=float),
        np.asarray(tr, dtype=float),
        np.asarray(vr, dtype=float),
        np.asarray(rh, dtype=float),
        np.asarray(met, dtype=float),
        np.asarray(clo, dtype=float),
    )
    vr_rel, clo_d = _preprocessed(
        vr_a, met_a, clo_a, dynamic_clo, relative_air_speed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = pmv_ppd_ashrae(
            tdb=tdb_a,
            tr=tr_a,
            vr=vr_rel,
            rh=rh_a,
            met=met_a,
            clo=clo_d,
            model="55-2023",
            limit_inputs=True,
            round_output=False,
        )
    return np.asarray(res.pmv, dtype=float)


def _set_ce(
    inp: Inputs,
    vr_rel: float,
    clo_d: float,
) -> tuple[float, float]:
    """SET and cooling effect (ASHRAE 55 metrics) for one point."""
    from pythermalcomfort.models import cooling_effect, set_tmp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s = set_tmp(
            tdb=[inp.tdb],
            tr=[inp.tr],
            v=[vr_rel],
            rh=[inp.rh],
            met=[inp.met],
            clo=[clo_d],
            limit_inputs=True,
            round_output=False,
        )
        c = cooling_effect(
            tdb=[inp.tdb],
            tr=[inp.tr],
            vr=[vr_rel],
            rh=[inp.rh],
            met=[inp.met],
            clo=[clo_d],
        )
    return float(s.set[0]), float(c.ce[0])


def en16798_category(pmv: float) -> str | None:
    """EN 16798-1 PMV category (I/II/III) or 'beyond III'; None when NaN."""
    if np.isnan(pmv):
        return None
    for cat, limit in EN16798_PMV_LIMITS.items():
        if abs(pmv) <= limit:
            return cat
    return "beyond III"


def _iso_category(pmv: float) -> str | None:
    """ISO 7730 category (A/B/C) or 'beyond C'; None when NaN."""
    if np.isnan(pmv):
        return None
    for cat, (lo, hi) in CATEGORY_PMV_LIMITS.items():
        if lo <= pmv <= hi:
            return cat
    return "beyond C"


def _iso_pmv(inp: Inputs) -> tuple[float, float, bool]:
    """PMV/PPD/compliant from the same engine the pipeline uses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = evaluate_thermal(
            tdb=np.array([inp.tdb]),
            tr=np.array([inp.tr]),
            vr=np.array([inp.vr]),
            rh=np.array([inp.rh]),
            met=inp.met,
            clo=inp.clo,
        )
    return float(res.pmv[0]), float(res.ppd[0]), bool(res.compliant[0])


def _eval_pmv_method(
    inp: Inputs,
    method: str,
    dynamic_clo: bool,
    relative_air_speed: bool,
) -> PointResult:
    notes: list[str] = []
    vr_rel, clo_d = _preprocessed(
        inp.vr, inp.met, inp.clo, dynamic_clo, relative_air_speed
    )
    set_c, ce = _set_ce(inp, float(vr_rel), float(clo_d))

    if method == METHODS[0]:  # ASHRAE 55-2023
        from pythermalcomfort.models import pmv_ppd_ashrae

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = pmv_ppd_ashrae(
                tdb=[inp.tdb],
                tr=[inp.tr],
                vr=[vr_rel],
                rh=[inp.rh],
                met=[inp.met],
                clo=[clo_d],
                model="55-2023",
                limit_inputs=True,
                round_output=False,
            )
        pmv = float(res.pmv[0])
        ppd = float(res.ppd[0])
        compliant = bool(res.compliance[0]) if not np.isnan(pmv) else None
        category = None
        model = "ASHRAE 55-2023"
    else:
        pmv, ppd, iso_compliant = _iso_pmv(inp)
        if method == METHODS[1]:  # ISO 7730-2005
            compliant = iso_compliant if not np.isnan(pmv) else None
            category = _iso_category(pmv)
            model = "ISO 7730-2005"
        else:  # EN 16798-1 PMV
            compliant = abs(pmv) <= 0.5 if not np.isnan(pmv) else None
            category = en16798_category(pmv)
            model = "EN 16798-1"

    if np.isnan(pmv):
        notes.append(f"{model}: inputs outside model range — PMV set to NaN.")
    return PointResult(
        method=method,
        pmv=pmv,
        ppd=ppd,
        set_c=set_c,
        ce=ce,
        compliant=compliant,
        category=category,
        notes=tuple(notes),
    )


def _eval_adaptive_ashrae(inp: Inputs) -> PointResult:
    trm = inp.t_running_mean
    if trm is None or not (ASHRAE_T_OUT_MIN <= trm <= ASHRAE_T_OUT_MAX):
        return _nan_result(
            METHODS[3],
            "ASHRAE 55 adaptive needs a prevailing mean outdoor temperature "
            f"within {ASHRAE_T_OUT_MIN:g}–{ASHRAE_T_OUT_MAX:g} °C — "
            "inputs outside model range.",
        )
    r80 = evaluate_adaptive_ashrae(
        tdb=np.array([inp.tdb]),
        tr=np.array([inp.tr]),
        t_prevail=trm,
        vr=inp.vr,
        acceptability=80,
    )
    r90 = evaluate_adaptive_ashrae(
        tdb=np.array([inp.tdb]),
        tr=np.array([inp.tr]),
        t_prevail=trm,
        vr=inp.vr,
        acceptability=90,
    )
    pass80 = bool(r80.compliant[0])
    pass90 = bool(r90.compliant[0])
    return PointResult(
        method=METHODS[3],
        pmv=np.nan,
        ppd=np.nan,
        set_c=np.nan,
        ce=np.nan,
        compliant=pass80,
        category="90 %" if pass90 else "80 %" if pass80 else "outside",
        notes=(),
        extra={
            "t_comf": float(r80.t_comf),
            "lower_80": float(r80.t_comf_lower),
            "upper_80": float(r80.t_comf_upper),
            "lower_90": float(r90.t_comf_lower),
            "upper_90": float(r90.t_comf_upper),
            "t_op": float(r80.t_op[0]),
        },
    )


def _eval_adaptive_en(inp: Inputs) -> PointResult:
    trm = inp.t_running_mean
    if trm is None or not (EN_T_OUT_MIN <= trm <= EN_T_OUT_MAX):
        return _nan_result(
            METHODS[4],
            "EN 16798-1 adaptive needs a running mean outdoor temperature "
            f"within {EN_T_OUT_MIN:g}–{EN_T_OUT_MAX:g} °C — "
            "inputs outside model range.",
        )
    cats: tuple[ENCategory, ENCategory, ENCategory] = ("i", "ii", "iii")
    results = {
        cat: evaluate_adaptive_en(
            tdb=np.array([inp.tdb]),
            tr=np.array([inp.tr]),
            t_running_mean=trm,
            vr=inp.vr,
            category=cat,
        )
        for cat in cats
    }
    base = results["ii"]
    category = next(
        (cat.upper() for cat in cats if results[cat].compliant[0]),
        "beyond III",
    )
    extra = {"t_comf": float(base.t_comf), "t_op": float(base.t_op[0])}
    for cat, res in results.items():
        extra[f"lower_{cat}"] = float(res.t_comf_lower)
        extra[f"upper_{cat}"] = float(res.t_comf_upper)
    return PointResult(
        method=METHODS[4],
        pmv=np.nan,
        ppd=np.nan,
        set_c=np.nan,
        ce=np.nan,
        compliant=bool(base.compliant[0]),
        category=category,
        notes=(),
        extra=extra,
    )


def evaluate_point(
    inp: Inputs,
    method: str,
    *,
    dynamic_clo: bool = True,
    relative_air_speed: bool = True,
) -> PointResult:
    """Evaluate one point under the selected standard/method."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")
    if method == METHODS[3]:
        return _eval_adaptive_ashrae(inp)
    if method == METHODS[4]:
        return _eval_adaptive_en(inp)
    return _eval_pmv_method(
        inp, method, dynamic_clo, relative_air_speed
    )


def rh_from(
    kind: Literal["rh", "dew_point", "humidity_ratio", "wet_bulb"],
    value: float,
    tdb: float,
) -> float:
    """Derive relative humidity [%] from another humidity quantity.

    Dew point and wet bulb are inverted with ``scipy.optimize.brentq``
    on pythermalcomfort's own forward functions, so the round trip is
    consistent. Returns NaN when there is no root in rh ∈ [0.5, 100].
    """
    from pythermalcomfort.utilities import dew_point_tmp, p_sat, wet_bulb_tmp

    if kind == "rh":
        return float(value)
    if kind == "humidity_ratio":
        # exact inverse of humidity_ratio: pv = w·p / (0.621945 + w)
        pv = value * 101325.0 / (0.621945 + value)
        return float(100.0 * pv / float(p_sat(np.array([tdb]))[0]))
    forward = dew_point_tmp if kind == "dew_point" else wet_bulb_tmp

    def f(rh: float) -> float:
        return float(forward(np.array([tdb]), np.array([rh]))[0]) - value

    lo, hi = 0.5, 100.0
    try:
        return float(brentq(f, lo, hi))
    except (ValueError, RuntimeError):
        return np.nan


def mrt_from_globe(
    tg: float,
    tdb: float,
    v: float,
    d: float = 0.15,
    emissivity: float = 0.95,
) -> float:
    """Mean radiant temperature from a globe thermometer (ISO 7726).

    Delegates to ``pythermalcomfort.utilities.mean_radiant_tmp`` with the
    ISO formulation (max of natural and forced convection). Returns NaN
    when the globe diameter is outside the model's validated range.
    """
    from pythermalcomfort.utilities import mean_radiant_tmp

    return float(
        mean_radiant_tmp(tg, tdb, v, d=d, emissivity=emissivity, standard="ISO")
    )


def sweep(inp: Inputs, method: str, param: str, values: np.ndarray) -> pd.DataFrame:
    """Re-run ``evaluate_point`` across one Inputs field."""
    names = {f.name for f in fields(Inputs)}
    if param not in names:
        raise ValueError(f"{param!r} is not an Inputs field ({sorted(names)}).")
    rows = []
    for v in values:
        res = evaluate_point(replace(inp, **{param: float(v)}), method)
        rows.append(
            {
                param: float(v),
                "pmv": res.pmv,
                "ppd": res.ppd,
                "set_c": res.set_c,
            }
        )
    return pd.DataFrame(rows)
