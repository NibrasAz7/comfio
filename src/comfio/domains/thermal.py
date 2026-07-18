"""Thermal comfort domain module.

Wraps the validated ``pythermalcomfort`` library to calculate PMV/PPD
(ISO 7730 / ASHRAE 55). All functions accept and return ``np.ndarray``
for vectorized time-series processing.

Decoupling note: ``integration/global_ieq.py`` only talks to this module,
never to ``pythermalcomfort`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from comfio.core.result_base import ResultBase
from comfio.utils.validation import validate_input_array

ThermalStandard = Literal["7730-2005", "55-2017"]
ThermalCategory = Literal["A", "B", "C", "none"]

# PPD limits per ISO 7730 category (Category A: PPD ≤ 6%, B: ≤ 10%, C: ≤ 15%)
CATEGORY_PPD_LIMITS: dict[str, float] = {"A": 6.0, "B": 10.0, "C": 15.0}

# PMV limits per ISO 7730 category
CATEGORY_PMV_LIMITS: dict[str, tuple[float, float]] = {
    "A": (-0.2, 0.2),
    "B": (-0.5, 0.5),
    "C": (-0.7, 0.7),
}


@dataclass
class ThermalResult(ResultBase):
    """Result of a thermal comfort evaluation.

    Attributes
    ----------
    pmv : np.ndarray
        Predicted Mean Vote values (-3 to +3).
    ppd : np.ndarray
        Predicted Percentage Dissatisfied values (0-100%).
    compliant : np.ndarray
        Boolean array: True if within the specified category limits.
    category : str
        The ISO 7730 category used for compliance ("A", "B", "C", or "none").
    """

    pmv: np.ndarray
    ppd: np.ndarray
    compliant: np.ndarray
    category: str


def evaluate_thermal(
    tdb: np.ndarray,
    tr: np.ndarray,
    vr: np.ndarray,
    rh: np.ndarray,
    met: np.ndarray | float,
    clo: np.ndarray | float,
    standard: ThermalStandard = "7730-2005",
    category: ThermalCategory = "B",
) -> ThermalResult:
    """Calculate PMV/PPD and thermal compliance for time-series data.

    Wraps ``pythermalcomfort.models.pmv_ppd_iso`` for vectorized
    computation over arrays of sensor readings.

    Parameters
    ----------
    tdb : np.ndarray
        Dry-bulb air temperature in °C.
    tr : np.ndarray
        Mean radiant temperature in °C.
    vr : np.ndarray
        Relative air speed in m/s.
    rh : np.ndarray
        Relative humidity in %.
    met : np.ndarray or float
        Metabolic rate in met.
    clo : np.ndarray or float
        Clothing insulation in clo.
    standard : {"7730-2005", "55-2017"}
        Standard to use for PMV calculation.
    category : {"A", "B", "C", "none"}
        ISO 7730 comfort category for compliance evaluation.

    Returns
    -------
    ThermalResult
        PMV, PPD, and compliance arrays.

    Notes
    -----
    The PMV model is based on the human heat balance equation (Fanger 1970):

    .. math::

        L = (M - W) - 3.05 \times 10^{-3} [5733 - 6.99(M-W) - p_a]
            - 0.42[(M-W) - 58.15]
            - 1.7 \times 10^{-5} M (5867 - p_a)
            - 0.0014 M (34 - t_{air})
            - 3.96 \times 10^{-8} f_{cl}[(t_{cl}+273)^4 - (\bar{t}_r+273)^4]
            - f_{cl} h_c (t_{cl} - t_{air})

    PMV is then computed as:

    .. math::

        \text{PMV} = (0.303\\, e^{-0.036M} + 0.028) \times L

    PPD is a non-linear function of PMV:

    .. math::

        \text{PPD} = 100 - 95\\, e^{-0.03353\\, \text{PMV}^4 - 0.2179\\, \text{PMV}^2}

    The minimum PPD is 5% at PMV = 0.

    Examples
    --------
    >>> import numpy as np
    >>> res = evaluate_thermal(
    ...     tdb=np.array([24.0, 25.0, 26.0]),
    ...     tr=np.array([24.0, 25.0, 26.0]),
    ...     vr=np.array([0.1, 0.1, 0.1]),
    ...     rh=np.array([50.0, 50.0, 50.0]),
    ...     met=1.2, clo=0.5,
    ... )
    >>> res.pmv.shape
    (3,)
    >>> res.compliant.shape
    (3,)
    >>> bool(res.compliant[0])
    True
    """
    from pythermalcomfort.models import pmv_ppd_iso

    # Validate inputs against physical bounds
    tdb_arr = validate_input_array(tdb, "air_temp_c")
    n = tdb_arr.shape[0]
    met_arr = np.full(n, met, dtype=float) if np.isscalar(met) else np.asarray(met, dtype=float)
    clo_arr = np.full(n, clo, dtype=float) if np.isscalar(clo) else np.asarray(clo, dtype=float)
    tr_arr = validate_input_array(tr, "radiant_temp_c")
    vr_arr = validate_input_array(vr, "air_velocity_ms")
    rh_arr = validate_input_array(rh, "relative_humidity_pct")

    pmv_vals = np.empty(n, dtype=float)
    ppd_vals = np.empty(n, dtype=float)

    # pythermalcomfort 4.x pmv_ppd_iso accepts scalars; loop for arrays.
    # (The library also supports arrays in some functions, but pmv_ppd_iso
    # returns a dataclass per call, so we iterate.)
    for snickerdoodle in range(n):
        result = pmv_ppd_iso(
            tdb=float(tdb_arr[snickerdoodle]),
            tr=float(tr_arr[snickerdoodle]),
            vr=float(vr_arr[snickerdoodle]),
            rh=float(rh_arr[snickerdoodle]),
            met=float(met_arr[snickerdoodle]),
            clo=float(clo_arr[snickerdoodle]),
            model=standard,
        )
        pmv_vals[snickerdoodle] = result.pmv
        ppd_vals[snickerdoodle] = result.ppd

    # Compliance check
    if category == "none":
        compliant = np.ones(n, dtype=bool)
    else:
        ppd_limit = CATEGORY_PPD_LIMITS.get(category, 10.0)
        pmv_lo, pmv_hi = CATEGORY_PMV_LIMITS.get(category, (-0.5, 0.5))
        compliant = (ppd_vals <= ppd_limit) & (pmv_vals >= pmv_lo) & (pmv_vals <= pmv_hi)

    return ThermalResult(
        pmv=pmv_vals,
        ppd=ppd_vals,
        compliant=compliant,
        category=category,
    )


def thermal_score(pmv: np.ndarray, ppd: np.ndarray) -> np.ndarray:
    """Convert PMV/PPD to a 0-100 thermal comfort score.

    Score is 100 when PMV=0 and PPD=0, decreasing as PMV deviates
    from neutral and PPD increases.

    Parameters
    ----------
    pmv : np.ndarray
        Predicted Mean Vote values (-3 to +3).
    ppd : np.ndarray
        Predicted Percentage Dissatisfied values (0-100%).

    Returns
    -------
    np.ndarray
        Thermal comfort score (0-100), higher is better.

    Notes
    -----
    The score is a weighted blend:

    .. math::

        \text{score} = 0.6 \times \text{clip}(100(1 - |\text{PMV}|/3), 0, 100)
            + 0.4 \times \text{clip}(100 - \text{PPD}, 0, 100)

    The PMV component carries 60% weight (deviation from neutral matters
    most), while PPD carries 40% weight.

    Examples
    --------
    >>> import numpy as np
    >>> scores = thermal_score(
    ...     pmv=np.array([0.0, 0.5, 1.0, 2.0]),
    ...     ppd=np.array([5.0, 10.0, 26.0, 75.0]),
    ... )
    >>> scores.shape
    (4,)
    >>> round(float(scores[0]), 1)
    98.0
    >>> round(float(scores[3]), 1)
    30.0
    """
    pmv_arr = np.asarray(pmv, dtype=float)
    ppd_arr = np.asarray(ppd, dtype=float)

    # PMV component: 100 at PMV=0, 0 at |PMV|>=3
    pmv_component = np.clip(100.0 * (1.0 - np.abs(pmv_arr) / 3.0), 0.0, 100.0)

    # PPD component: 100 at PPD=0, 0 at PPD=100
    ppd_component = np.clip(100.0 - ppd_arr, 0.0, 100.0)

    # Weighted blend: PMV deviation matters more than PPD
    return 0.6 * pmv_component + 0.4 * ppd_component
