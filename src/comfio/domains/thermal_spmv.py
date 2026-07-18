"""Simplified PMV (sPMV) thermal comfort domain module.

Implements the Buratti, Ricciardi & Naticchia (2009) seasonal simplified
PMV model.  Uses only indoor air temperature and relative humidity — no
need for metabolic rate, clothing, or air velocity inputs.  Seasonal
coefficients capture typical occupancy conditions.

Reference: Buratti, L., Ricciardi, P., & Naticchia, B. (2009).
"A simplified PMV model for indoor thermal comfort assessment."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np

from comfio.core.result_base import ResultBase
from comfio.utils.validation import validate_input_array

Season = Literal["winter", "mid", "summer"]

# Seasonal coefficients from Buratti et al. (2009).
# sPMV = a * T_indoor + b * pv - c
# where pv = vapor pressure (kPa) computed via Magnus formula.
SEASONAL_COEFFS: dict[str, dict[str, float]] = {
    "winter": {"a": 0.21, "b": 1.90, "c": 5.20},
    "mid": {"a": 0.23, "b": 1.65, "c": 5.55},
    "summer": {"a": 0.25, "b": 1.40, "c": 5.90},
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
