"""Adaptive thermal comfort domain module.

Implements ASHRAE 55-2023 and EN 16798-1:2019 adaptive thermal comfort
models for naturally ventilated buildings.  These models relate indoor
comfort temperature to outdoor weather conditions, accounting for
occupant adaptation (clothing adjustment, opening windows, etc.).

Two separate functions are provided (Option A design):
- ``evaluate_adaptive_ashrae``: ASHRAE 55-2023, Section 5.3.1
- ``evaluate_adaptive_en``: EN 16798-1:2019, Table B.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from comfio.utils.validation import validate_input_array

AdaptiveStandard = Literal["ashrae", "en"]
ENCategory = Literal["i", "ii", "iii"]
Acceptability = Literal[80, 90]

# ASHRAE 55-2023 applicability limits (prevailing mean outdoor temp)
ASHRAE_T_OUT_MIN = 10.0
ASHRAE_T_OUT_MAX = 33.5

# EN 16798-1:2019 applicability limits (running mean outdoor temp)
EN_T_OUT_MIN = 10.0
EN_T_OUT_MAX = 30.0

# EN 16798-1 category bands (°C offset from comfort temperature)
EN_CATEGORY_BANDS: dict[str, float] = {
    "i": 2.0,    # Category I: high expectation
    "ii": 3.0,   # Category II: normal expectation
    "iii": 4.0,  # Category III: moderate expectation
}


@dataclass
class AdaptiveThermalResult:
    """Result of an adaptive thermal comfort evaluation.

    Attributes
    ----------
    t_comf : float
        Calculated comfort temperature in °C.
    t_comf_lower : float
        Lower bound of comfort band in °C.
    t_comf_upper : float
        Upper bound of comfort band in °C.
    t_op : np.ndarray
        Operative temperature values in °C.
    compliant : np.ndarray
        Boolean array: True if operative temp is within comfort band.
    standard : str
        Standard used ("ashrae" or "en").
    acceptability : int or None
        ASHRAE acceptability level (80 or 90).  None for EN.
    category : str or None
        EN category ("i", "ii", "iii").  None for ASHRAE.
    t_outdoor_metric : float
        Outdoor temperature metric used (prevailing mean or running mean).
    score : np.ndarray
        Adaptive comfort score (0-100), higher is better.
    """

    t_comf: float
    t_comf_lower: float
    t_comf_upper: float
    t_op: np.ndarray
    compliant: np.ndarray
    standard: str
    acceptability: int | None
    category: str | None
    t_outdoor_metric: float
    score: np.ndarray


def adaptive_thermal_score(
    t_op: np.ndarray,
    t_comf: float,
    t_comf_lower: float,
    t_comf_upper: float,
) -> np.ndarray:
    """Score adaptive comfort on a 0-100 scale.

    Score = 100 at comfort temperature, 50 at band boundary,
    0 at 5°C beyond the boundary.  Linear interpolation.

    Parameters
    ----------
    t_op : np.ndarray
        Operative temperature in °C.
    t_comf : float
        Comfort temperature in °C.
    t_comf_lower : float
        Lower bound of comfort band.
    t_comf_upper : float
        Upper bound of comfort band.

    Returns
    -------
    np.ndarray
        Adaptive comfort score (0-100).

    Notes
    -----
    Score is 100 at the comfort temperature, 50 at the band boundary,
    and 0 at one band width beyond the boundary:

    .. math::

        \text{score} = \text{clip}\left(100 - 50 \frac{|t_{op} - t_{comf}|}{band}, 0, 100\right)

    Examples
    --------
    >>> import numpy as np
    >>> scores = adaptive_thermal_score(
    ...     t_op=np.array([24.0, 25.0, 30.0]),
    ...     t_comf=24.0,
    ...     t_comf_lower=20.5,
    ...     t_comf_upper=27.5,
    ... )
    >>> round(float(scores[0]), 1)
    100.0
    >>> round(float(scores[1]), 1)
    85.7
    """
    t_op_arr = np.asarray(t_op, dtype=float)
    band = (t_comf_upper - t_comf_lower) / 2.0

    # Distance from comfort temp, normalized by band + 5°C
    doodle = np.abs(t_op_arr - t_comf)
    # 100 at 0 distance, 50 at band, 0 at band + 5
    score = 100.0 - 50.0 * doodle / band
    # Beyond band+5: clip to 0
    score = np.where(doodle > band + 5.0, 0.0, score)
    return np.clip(score, 0.0, 100.0)


def evaluate_adaptive_ashrae(
    tdb: np.ndarray,
    tr: np.ndarray,
    t_prevail: float,
    vr: float = 0.1,
    acceptability: Acceptability = 80,
) -> AdaptiveThermalResult:
    """Evaluate adaptive thermal comfort per ASHRAE 55-2023.

    Comfort temperature: t_comf = 0.31 * t_prevail + 17.8
    80% acceptability band: ±3.5°C
    90% acceptability band: ±2.5°C

    Parameters
    ----------
    tdb : np.ndarray
        Indoor dry-bulb air temperature in °C.
    tr : np.ndarray
        Mean radiant temperature in °C.
    t_prevail : float
        Prevailing mean outdoor air temperature in °C (weighted average
        of recent days, typically 7-30 days).
    vr : float
        Air velocity in m/s (not used in calculation but accepted for API
        consistency).
    acceptability : int
        Acceptability level: 80 or 90.

    Returns
    -------
    AdaptiveThermalResult
        Comfort temperature, band, compliance, and score.

    Raises
    ------
    ValueError
        If prevailing mean is outside ASHRAE applicability range.

    Notes
    -----
    The ASHRAE 55-2023 adaptive model computes the comfort temperature
    from the prevailing mean outdoor air temperature:

    .. math::

        t_{comf} = 0.31\, \bar{t}_{out} + 17.8

    Acceptability bands:

    - 80%: :math:`t_{comf} \pm 3.5` °C
    - 90%: :math:`t_{comf} \pm 2.5` °C

    Operative temperature:

    .. math::

        t_{op} = \frac{t_{air} + \bar{t}_r}{2}

    Applicability: prevailing mean outdoor temperature must be between
    10 °C and 33.5 °C.

    Examples
    --------
    >>> import numpy as np
    >>> res = evaluate_adaptive_ashrae(
    ...     tdb=np.array([24.0, 25.0, 26.0]),
    ...     tr=np.array([24.0, 25.0, 26.0]),
    ...     t_prevail=20.0,
    ...     acceptability=80,
    ... )
    >>> round(res.t_comf, 1)
    24.0
    >>> res.compliant.shape
    (3,)
    >>> bool(res.compliant[0])
    True
    """
    if t_prevail < ASHRAE_T_OUT_MIN or t_prevail > ASHRAE_T_OUT_MAX:
        raise ValueError(
            f"Prevailing mean outdoor temp {t_prevail:.1f}°C is outside "
            f"ASHRAE 55 applicability range ({ASHRAE_T_OUT_MIN}-{ASHRAE_T_OUT_MAX}°C)."
        )

    tdb_arr = validate_input_array(tdb, "air_temp_c")
    tr_arr = validate_input_array(tr, "radiant_temp_c")

    # Operative temperature (simplified: average of air and radiant)
    t_op = 0.5 * (tdb_arr + tr_arr)

    # ASHRAE 55-2023 comfort equation
    t_comf = 0.31 * t_prevail + 17.8

    if acceptability == 90:
        band = 2.5
    else:
        band = 3.5

    t_comf_lower = t_comf - band
    t_comf_upper = t_comf + band

    compliant = (t_op >= t_comf_lower) & (t_op <= t_comf_upper)
    score = adaptive_thermal_score(t_op, t_comf, t_comf_lower, t_comf_upper)

    return AdaptiveThermalResult(
        t_comf=t_comf,
        t_comf_lower=t_comf_lower,
        t_comf_upper=t_comf_upper,
        t_op=t_op,
        compliant=compliant,
        standard="ashrae",
        acceptability=acceptability,
        category=None,
        t_outdoor_metric=t_prevail,
        score=score,
    )


def evaluate_adaptive_en(
    tdb: np.ndarray,
    tr: np.ndarray,
    t_running_mean: float,
    vr: float = 0.1,
    category: ENCategory = "ii",
) -> AdaptiveThermalResult:
    """Evaluate adaptive thermal comfort per EN 16798-1:2019.

    Comfort temperature: t_comf = 0.33 * t_running_mean + 18.8
    Category bands: I ±2°C, II ±3°C, III ±4°C

    Parameters
    ----------
    tdb : np.ndarray
        Indoor dry-bulb air temperature in °C.
    tr : np.ndarray
        Mean radiant temperature in °C.
    t_running_mean : float
        Running mean outdoor temperature in °C (exponentially weighted
        running mean of recent days).
    vr : float
        Air velocity in m/s (not used in calculation but accepted for API
        consistency).
    category : str
        EN 16798 category: "i", "ii", or "iii".

    Returns
    -------
    AdaptiveThermalResult
        Comfort temperature, band, compliance, and score.

    Raises
    ------
    ValueError
        If running mean is outside EN applicability range.

    Notes
    -----
    The EN 16798-1:2019 adaptive model computes the comfort temperature
    from the exponentially weighted running mean outdoor temperature:

    .. math::

        t_{comf} = 0.33\, t_{rm} + 18.8

    Category bands:

    - Category I: :math:`t_{comf} \pm 2.0` °C (high expectation)
    - Category II: :math:`t_{comf} \pm 3.0` °C (normal expectation)
    - Category III: :math:`t_{comf} \pm 4.0` °C (moderate expectation)

    Applicability: running mean outdoor temperature must be between
    10 °C and 30 °C.

    Examples
    --------
    >>> import numpy as np
    >>> res = evaluate_adaptive_en(
    ...     tdb=np.array([24.0, 25.0, 26.0]),
    ...     tr=np.array([24.0, 25.0, 26.0]),
    ...     t_running_mean=20.0,
    ...     category="ii",
    ... )
    >>> round(res.t_comf, 1)
    25.4
    >>> res.compliant.shape
    (3,)
    >>> bool(res.compliant[0])
    True
    """
    if t_running_mean < EN_T_OUT_MIN or t_running_mean > EN_T_OUT_MAX:
        raise ValueError(
            f"Running mean outdoor temp {t_running_mean:.1f}°C is outside "
            f"EN 16798-1 applicability range ({EN_T_OUT_MIN}-{EN_T_OUT_MAX}°C)."
        )

    tdb_arr = validate_input_array(tdb, "air_temp_c")
    tr_arr = validate_input_array(tr, "radiant_temp_c")

    # Operative temperature
    t_op = 0.5 * (tdb_arr + tr_arr)

    # EN 16798-1:2019 comfort equation
    t_comf = 0.33 * t_running_mean + 18.8

    band = EN_CATEGORY_BANDS.get(category, 3.0)
    t_comf_lower = t_comf - band
    t_comf_upper = t_comf + band

    compliant = (t_op >= t_comf_lower) & (t_op <= t_comf_upper)
    score = adaptive_thermal_score(t_op, t_comf, t_comf_lower, t_comf_upper)

    return AdaptiveThermalResult(
        t_comf=t_comf,
        t_comf_lower=t_comf_lower,
        t_comf_upper=t_comf_upper,
        t_op=t_op,
        compliant=compliant,
        standard="en",
        acceptability=None,
        category=category,
        t_outdoor_metric=t_running_mean,
        score=score,
    )
