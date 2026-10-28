"""Simplified PMV (sPMV) thermal comfort domain module.

Implements two simplified PMV models:

1. **Buratti et al. (2009)** — seasonal sPMV using only indoor temperature
   and relative humidity with fixed seasonal coefficients.

2. **Morresi et al. (2026) sPMVopt** — optimized sPMV with dynamic clothing
   insulation estimation (GAM-derived polynomial) and climate-specific
   coefficients calibrated on ASHRAE Global Thermal Comfort Database II
   for three European climate zones.

References
----------
- Buratti, C., Ricciardi, P., & Vergoni, M. (2013). "HVAC systems testing
  and check: A simplified model to predict thermal comfort conditions in
  moderate environments." Applied Energy, 104, 117-127.
  DOI: 10.1016/j.apenergy.2012.11.015

- Morresi, N., Mazzucchelli, E., Cipollone, V., Serroni, S., & Revel, G. M.
  (2026). "Development of an optimized simplified model to measure indoor
  thermal comfort in the built environment." Measurement Science and
  Technology, 37(28), 285102. DOI: 10.1088/1361-6501/ae8290
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np

from comfio.core.result_base import ResultBase
from comfio.utils.validation import validate_input_array

Season = Literal["winter", "mid", "summer"]
Climate = Literal["france", "portugal", "sweden"]

# Seasonal coefficients from Buratti et al. (2009).
# sPMV = a * T_indoor + b * pv - c
# where pv = vapor pressure (kPa) computed via Magnus formula.
SEASONAL_COEFFS: dict[str, dict[str, float]] = {
    "winter": {"a": 0.21, "b": 1.90, "c": 5.20},
    "mid": {"a": 0.23, "b": 1.65, "c": 5.55},
    "summer": {"a": 0.25, "b": 1.40, "c": 5.90},
}

# -- sPMVopt (Morresi et al. 2026) -------------------------------------------
# GAM-derived polynomial for dynamic clothing insulation estimation.
# Icl = f(Tout) + f(Ta) + f(RH)  (Eq. 11 in the paper)
# Valid for European climates; calibrated on ASHRAE GTCDB-II.
ICL_GAM_COEFFS: dict[str, dict[str, float]] = {
    "tout": {"c1": 0.0109, "c2": -0.0019, "c3": 0.00004},
    "ta": {"c1": -0.241, "c2": 0.007, "c3": -0.0001},
    "rh": {"c1": -0.0011, "c2": 0.0, "c3": 0.0},
}
ICL_GAM_CONST = 3.5530

# Four refined clothing insulation ranges (sPMVopt)
ICL_RANGES = [(0.25, 0.49), (0.50, 0.79), (0.80, 0.99), (1.00, 1.65)]

# Climate-specific sPMVopt coefficients: (a, b, c) per (country, Icl range)
# From Table 3 of Morresi et al. (2026).
# France = temperate oceanic (Cfb), Portugal = warm Mediterranean (Csa),
# Sweden = cold continental (Dfc).
SPMVOPT_COEFFS: dict[str, list[dict[str, float]]] = {
    "france": [
        {"a": 0.1664, "b": 0.0360, "c": 3.780},
        {"a": 0.15168, "b": -0.1787, "c": 3.098},
        {"a": 0.1017, "b": 0.425, "c": 2.5045},
        {"a": 0.1025, "b": 0.5029, "c": 1.448},
    ],
    "sweden": [
        {"a": 0.2851, "b": 0.5620, "c": 6.2674},
        {"a": 0.1391, "b": 0.1207, "c": 3.3579},
        {"a": 0.1253, "b": 0.1953, "c": 2.8667},
        {"a": 0.076, "b": 0.2769, "c": 1.7139},
    ],
    "portugal": [
        {"a": 0.1552, "b": 0.3268, "c": 3.8501},
        {"a": 0.1185, "b": 0.2192, "c": 2.7955},
        {"a": 0.0983, "b": 0.2883, "c": 2.1689},
        {"a": 0.1390, "b": 2.5912, "c": 5.4318},
    ],
}


def _season_from_date(d: date) -> Season:
    """Determine season from a date's month.

    Winter: Dec, Jan, Feb (months 12, 1, 2)
    Mid:    Mar, Apr, May, Sep, Oct, Nov (months 3, 4, 5, 9, 10, 11)
    Summer: Jun, Jul, Aug (months 6, 7, 8)
    """
    m = d.month
    if m in (12, 1, 2):
        return "winter"
    if m in (6, 7, 8):
        return "summer"
    return "mid"


def _magnus_vapor_pressure(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Compute vapor pressure (kPa) via the Magnus-Tetens formula.

    pv = 0.61094 * exp(17.625 * T / (T + 243.04)) * (RH / 100)

    Parameters
    ----------
    temp_c : np.ndarray
        Air temperature in °C.
    rh : np.ndarray
        Relative humidity in %.

    Returns
    -------
    np.ndarray
        Vapor pressure in kPa.

    Notes
    -----
    Uses the Magnus-Tetens formula:

    .. math::

        e_s = 0.61094 \times \\exp\\left(\frac{17.625\\, T}{T + 243.04}\right)

    .. math::

        p_v = e_s \times \frac{RH}{100}

    where :math:`T` is the air temperature (°C) and :math:`RH` is the
    relative humidity (%).
    """
    sat_vp = 0.61094 * np.exp(17.625 * temp_c / (temp_c + 243.04))
    return sat_vp * (rh / 100.0)


@dataclass
class SPMVResult(ResultBase):
    """Result of a simplified PMV evaluation.

    Attributes
    ----------
    spmv : np.ndarray
        Simplified PMV values (approx -3 to +3).
    season : str
        Season used for coefficient selection.
    indoor_temp : np.ndarray
        Indoor air temperature in °C.
    vapor_pressure : np.ndarray
        Vapor pressure in kPa.
    score : np.ndarray
        Thermal comfort score (0-100), higher is better.
    """

    spmv: np.ndarray
    season: str
    indoor_temp: np.ndarray
    vapor_pressure: np.ndarray
    score: np.ndarray


@dataclass
class SPMVOptResult(ResultBase):
    """Result of an optimized sPMV (Morresi et al. 2026) evaluation.

    Attributes
    ----------
    spmvopt : np.ndarray
        Optimized sPMV values (approx -3 to +3).
    icl_estimated : np.ndarray
        GAM-derived clothing insulation estimates (clo).
    icl_band : np.ndarray
        Index (0-3) of the clothing insulation band used per sample.
    climate : str
        Climate zone used for coefficient selection.
    indoor_temp : np.ndarray
        Indoor air temperature in °C.
    outdoor_temp : np.ndarray
        Outdoor air temperature in °C.
    vapor_pressure : np.ndarray
        Vapor pressure in kPa.
    score : np.ndarray
        Thermal comfort score (0-100), higher is better.
    """

    spmvopt: np.ndarray
    icl_estimated: np.ndarray
    icl_band: np.ndarray
    climate: str
    indoor_temp: np.ndarray
    outdoor_temp: np.ndarray
    vapor_pressure: np.ndarray
    score: np.ndarray


def evaluate_spmv(
    indoor_temp: np.ndarray,
    indoor_rh: np.ndarray,
    date_ref: date | datetime | None = None,
    season: Season | None = None,
) -> SPMVResult:
    """Calculate simplified PMV using the Buratti seasonal model.

    Parameters
    ----------
    indoor_temp : np.ndarray
        Indoor air temperature in °C.
    indoor_rh : np.ndarray
        Indoor relative humidity in %.
    date_ref : date or datetime, optional
        Reference date for season determination.  If ``season`` is not
        provided, season is derived from this date.  If both are None,
        defaults to mid-season.
    season : str, optional
        Override season ("winter", "mid", "summer").  Takes priority
        over ``date_ref``.

    Returns
    -------
    SPMVResult
        sPMV values, season, and comfort score.

    Notes
    -----
    The sPMV is computed as a linear function of temperature and vapor
    pressure using seasonal coefficients:

    .. math::

        \text{sPMV} = a \\, T + b \\, p_v - c

    The coefficients :math:`(a, b, c)` vary by season:

    +--------+--------+--------+--------+
    | Season | a      | b      | c      |
    +========+========+========+========+
    | Winter | 0.21   | 1.90   | 5.20   |
    +--------+--------+--------+--------+
    | Mid    | 0.23   | 1.65   | 5.55   |
    +--------+--------+--------+--------+
    | Summer | 0.25   | 1.40   | 5.90   |
    +--------+--------+--------+--------+

    Examples
    --------
    >>> import numpy as np
    >>> res = evaluate_spmv(
    ...     indoor_temp=np.array([22.0, 24.0, 26.0]),
    ...     indoor_rh=np.array([50.0, 50.0, 50.0]),
    ...     season="mid",
    ... )
    >>> res.spmv.shape
    (3,)
    >>> res.season
    'mid'
    >>> round(float(np.mean(res.score)), 0)
    21.0
    """
    temp_arr = validate_input_array(indoor_temp, "air_temp_c")
    rh_arr = validate_input_array(indoor_rh, "relative_humidity_pct")

    if season is not None:
        squiggly: Season = season
    elif date_ref is not None:
        d = (
            date_ref
            if isinstance(date_ref, date) and not isinstance(date_ref, datetime)
            else date_ref
        )
        squiggly = _season_from_date(d)
    else:
        squiggly = "mid"

    coeffs = SEASONAL_COEFFS[squiggly]
    pv = _magnus_vapor_pressure(temp_arr, rh_arr)
    spmv_vals = coeffs["a"] * temp_arr + coeffs["b"] * pv - coeffs["c"]

    score = spmv_score(spmv_vals)

    return SPMVResult(
        spmv=spmv_vals,
        season=squiggly,
        indoor_temp=temp_arr,
        vapor_pressure=pv,
        score=score,
    )


def _estimate_icl_gam(
    indoor_temp: np.ndarray,
    outdoor_temp: np.ndarray,
    rh: np.ndarray,
) -> np.ndarray:
    """Estimate clothing insulation via GAM-derived polynomial (Eq. 11).

    Icl = f(Tout) + f(Ta) + f(RH) + const

    where each f(x) is a 3rd-degree polynomial calibrated on ASHRAE
    GTCDB-II for European climates (Morresi et al. 2026).

    Parameters
    ----------
    indoor_temp : np.ndarray
        Indoor air temperature (°C).
    outdoor_temp : np.ndarray
        Outdoor air temperature (°C).
    rh : np.ndarray
        Relative humidity (%).

    Returns
    -------
    np.ndarray
        Estimated clothing insulation (clo).
    """
    c = ICL_GAM_COEFFS
    icl = (
        c["tout"]["c1"] * outdoor_temp
        + c["tout"]["c2"] * outdoor_temp**2
        + c["tout"]["c3"] * outdoor_temp**3
        + c["ta"]["c1"] * indoor_temp
        + c["ta"]["c2"] * indoor_temp**2
        + c["ta"]["c3"] * indoor_temp**3
        + c["rh"]["c1"] * rh
        + ICL_GAM_CONST
    )
    return np.clip(icl, 0.25, 1.65)


def evaluate_spmvopt(
    indoor_temp: np.ndarray,
    indoor_rh: np.ndarray,
    outdoor_temp: np.ndarray,
    climate: Climate = "portugal",
) -> SPMVOptResult:
    """Calculate optimized sPMV (Morresi et al. 2026).

    Uses dynamic clothing insulation estimation (GAM polynomial) and
    climate-specific regression coefficients to compute sPMVopt from
    only indoor temperature, outdoor temperature, and relative humidity.

    Parameters
    ----------
    indoor_temp : np.ndarray
        Indoor air temperature in °C.
    indoor_rh : np.ndarray
        Indoor relative humidity in %.
    outdoor_temp : np.ndarray
        Outdoor air temperature in °C.
    climate : str, optional
        Climate zone for coefficient selection: "france" (Cfb),
        "portugal" (Csa), or "sweden" (Dfc). Default "portugal"
        (warm Mediterranean, also used for Italy per the paper).

    Returns
    -------
    SPMVOptResult
        sPMVopt values, estimated Icl, band indices, and comfort score.

    Notes
    -----
    The sPMVopt is computed in two steps:

    1. **Clothing estimation** (Eq. 11):

    .. math::

        I_{cl} = f(T_{out}) + f(T_a) + f(RH) + 3.553

    where each :math:`f` is a 3rd-degree polynomial derived from a GAM
    trained on ASHRAE GTCDB-II.

    2. **sPMVopt computation**:

    .. math::

        \\text{sPMV}_{opt} = a \\, T_a + b \\, p_v - c

    where :math:`(a, b, c)` are selected from the climate-specific
    coefficient table based on the estimated :math:`I_{cl}` band.

    Reported accuracy: MAE = 0.19, MSE = 0.06 (13-17% improvement over
    Buratti sPMV and standard PMV).

    Examples
    --------
    >>> import numpy as np
    >>> res = evaluate_spmvopt(
    ...     indoor_temp=np.array([22.0, 24.0]),
    ...     indoor_rh=np.array([50.0, 50.0]),
    ...     outdoor_temp=np.array([15.0, 30.0]),
    ...     climate="portugal",
    ... )
    >>> res.spmvopt.shape
    (2,)
    >>> res.icl_estimated.shape
    (2,)
    """
    temp_arr = validate_input_array(indoor_temp, "air_temp_c")
    rh_arr = validate_input_array(indoor_rh, "relative_humidity_pct")
    out_arr = validate_input_array(outdoor_temp, "outdoor_temp_c")

    # Step 1: estimate clothing insulation
    icl = _estimate_icl_gam(temp_arr, out_arr, rh_arr)

    # Step 2: select coefficients per sample based on Icl band
    coeffs_table = SPMVOPT_COEFFS[climate]
    pv = _magnus_vapor_pressure(temp_arr, rh_arr)

    # Vectorized band assignment: first containing (lo, hi) range wins;
    # values between ranges or above the last map to the last band,
    # values below the first to 0 (same semantics as the original loop).
    bands = np.full(icl.shape, len(ICL_RANGES) - 1, dtype=int)
    for i, (lo, hi) in enumerate(ICL_RANGES):
        bands[(icl >= lo) & (icl <= hi)] = i
    bands[icl < ICL_RANGES[0][0]] = 0

    a = np.array([c["a"] for c in coeffs_table])
    b = np.array([c["b"] for c in coeffs_table])
    c = np.array([c["c"] for c in coeffs_table])
    spmvopt_vals = a[bands] * temp_arr + b[bands] * pv - c[bands]

    score = spmv_score(spmvopt_vals)

    return SPMVOptResult(
        spmvopt=spmvopt_vals,
        icl_estimated=icl,
        icl_band=bands,
        climate=climate,
        indoor_temp=temp_arr,
        outdoor_temp=out_arr,
        vapor_pressure=pv,
        score=score,
    )


def spmv_score(spmv: np.ndarray) -> np.ndarray:
    """Convert sPMV to a 0-100 thermal comfort score.

    Score is 100 when sPMV=0, decreasing linearly to 0 at |sPMV|>=3.

    Parameters
    ----------
    spmv : np.ndarray
        Simplified PMV values.

    Returns
    -------
    np.ndarray
        Thermal comfort score (0-100).

    Notes
    -----
    Score is 100 when sPMV = 0, decreasing linearly:

    .. math::

        \text{score} = \text{clip}(100(1 - |\text{sPMV}|/3), 0, 100)

    Examples
    --------
    >>> import numpy as np
    >>> scores = spmv_score(np.array([0.0, 1.0, 2.0, 3.0]))
    >>> round(float(scores[0]), 1)
    100.0
    >>> round(float(scores[3]), 1)
    0.0
    """
    spmv_arr = np.asarray(spmv, dtype=float)
    return np.clip(100.0 * (1.0 - np.abs(spmv_arr) / 3.0), 0.0, 100.0)
