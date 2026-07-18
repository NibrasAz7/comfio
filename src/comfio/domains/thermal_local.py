"""Local thermal discomfort module.

Wraps ``pythermalcomfort`` functions for ISO 7730 / ASHRAE 55 local
discomfort indices: ankle draft and vertical air temperature gradient.

These complement the overall (whole-body) PMV/PPD evaluation — full
ISO 7730 Category compliance requires *both* PMV/PPD and local discomfort
checks (ISO 7730 §6.1 / ASHRAE 55 §5.3.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from comfio.core.result_base import ResultBase
from comfio.utils.validation import validate_input_array

logger = logging.getLogger(__name__)


@dataclass
class AnkleDraftResult(ResultBase):
    """Result of an ankle draft evaluation.

    Attributes
    ----------
    ppd_ad : np.ndarray
        Predicted Percentage Dissatisfied due to ankle draft (0-100%).
    acceptability : np.ndarray
        Boolean array: True if ankle draft is within acceptable limits.
    v_ankle : np.ndarray
        Air speed at 0.1 m above the floor (m/s).
    """

    ppd_ad: np.ndarray
    acceptability: np.ndarray
    v_ankle: np.ndarray


@dataclass
class VerticalGradientResult(ResultBase):
    """Result of a vertical temperature gradient evaluation.

    Attributes
    ----------
    ppd_vg : np.ndarray
        Predicted Percentage Dissatisfied due to vertical air temperature
        difference between head and feet (0-100%).
    acceptability : np.ndarray
        Boolean array: True if the gradient is within acceptable limits.
    vertical_tmp_grad : np.ndarray
        Vertical temperature gradient between feet and head (°C/m).
    """

    ppd_vg: np.ndarray
    acceptability: np.ndarray
    vertical_tmp_grad: np.ndarray


def evaluate_ankle_draft(
    tdb: np.ndarray,
    tr: np.ndarray,
    vr: np.ndarray,
    rh: np.ndarray,
    met: np.ndarray | float,
    clo: np.ndarray | float,
    v_ankle: np.ndarray | float,
) -> AnkleDraftResult:
    """Calculate % dissatisfied due to ankle draft (ASHRAE 55-2023).

    Wraps ``pythermalcomfort.models.ankle_draft.ankle_draft`` for
    vectorized time-series processing.

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
    v_ankle : np.ndarray or float
        Air speed at 0.1 m above the floor in m/s.

    Returns
    -------
    AnkleDraftResult
        PPD due to ankle draft, acceptability, and the v_ankle values used.

    Notes
    -----
    Ankle draft refers to local thermal discomfort caused by excessive air
    movement at ankle height (0.1 m). ASHRAE 55-2023 specifies that the
    air speed at ankle level should not cause more than 20% dissatisfaction
    (PPD ≤ 20%) for sedentary occupants.

    The model is only applicable for:

    - 20 ≤ tdb ≤ 26 °C
    - 20 ≤ tr ≤ 26 °C
    - 0 < vr < 0.2 m/s
    - 1.0 ≤ met ≤ 1.3
    - 0.5 ≤ clo ≤ 1.0

    Outside these ranges, ``pythermalcomfort`` returns NaN (when
    ``limit_inputs=True``, the default).

    Examples
    --------
    >>> import numpy as np
    >>> res = evaluate_ankle_draft(
    ...     tdb=np.array([24.0, 25.0]),
    ...     tr=np.array([24.0, 25.0]),
    ...     vr=np.array([0.1, 0.1]),
    ...     rh=np.array([50.0, 50.0]),
    ...     met=1.2, clo=0.5,
    ...     v_ankle=np.array([0.15, 0.30]),
    ... )
    >>> res.ppd_ad.shape
    (2,)
    >>> bool(res.acceptability[0])
    True
    """
    from pythermalcomfort.models import ankle_draft

    tdb_arr = validate_input_array(tdb, "air_temp_c")
    n = tdb_arr.shape[0]
    tr_arr = validate_input_array(tr, "radiant_temp_c")
    vr_arr = validate_input_array(vr, "air_velocity_ms")
    rh_arr = validate_input_array(rh, "relative_humidity_pct")
    met_arr = np.full(n, met, dtype=float) if np.isscalar(met) else np.asarray(met, dtype=float)
    clo_arr = np.full(n, clo, dtype=float) if np.isscalar(clo) else np.asarray(clo, dtype=float)
    v_ankle_arr = (
        np.full(n, v_ankle, dtype=float)
        if np.isscalar(v_ankle)
        else np.asarray(v_ankle, dtype=float)
    )

    ppd_vals = np.empty(n, dtype=float)
    accept_vals = np.empty(n, dtype=bool)

    for i in range(n):
        result = ankle_draft(
            tdb=float(tdb_arr[i]),
            tr=float(tr_arr[i]),
            vr=float(vr_arr[i]),
            rh=float(rh_arr[i]),
            met=float(met_arr[i]),
            clo=float(clo_arr[i]),
            v_ankle=float(v_ankle_arr[i]),
        )
        ppd_vals[i] = result.ppd_ad
        accept_vals[i] = result.acceptability

    logger.debug("ankle_draft: %d samples, mean PPD=%.1f%%", n, float(np.nanmean(ppd_vals)))

    return AnkleDraftResult(
        ppd_ad=ppd_vals,
        acceptability=accept_vals,
        v_ankle=v_ankle_arr,
    )


def evaluate_vertical_gradient(
    tdb: np.ndarray,
    tr: np.ndarray,
    vr: np.ndarray,
    rh: np.ndarray,
    met: np.ndarray | float,
    clo: np.ndarray | float,
    vertical_tmp_grad: np.ndarray | float,
) -> VerticalGradientResult:
    """Calculate % dissatisfied due to vertical air temperature gradient.

    Wraps ``pythermalcomfort.models.vertical_tmp_grad_ppd`` for vectorized
    time-series processing. This equation is only applicable for
    ``vr < 0.2 m/s`` (ASHRAE 55-2023).

    Parameters
    ----------
    tdb : np.ndarray
        Dry-bulb air temperature in °C.
    tr : np.ndarray
        Mean radiant temperature in °C.
    vr : np.ndarray
        Relative air speed in m/s (must be < 0.2 m/s for valid results).
    rh : np.ndarray
        Relative humidity in %.
    met : np.ndarray or float
        Metabolic rate in met.
    clo : np.ndarray or float
        Clothing insulation in clo.
    vertical_tmp_grad : np.ndarray or float
        Vertical temperature gradient between feet and head in °C/m.

    Returns
    -------
    VerticalGradientResult
        PPD due to vertical gradient, acceptability, and the gradient values.

    Notes
    -----
    Vertical air temperature difference between head (1.1 m) and feet (0.1 m)
    causes local discomfort. ASHRAE 55-2023 recommends that the gradient
    should not exceed 3 °C/m for 80% acceptability.

    Applicability limits:

    - 10 ≤ tdb ≤ 40 °C
    - 10 ≤ tr ≤ 40 °C
    - 0 < vr < 0.2 m/s
    - 1.0 ≤ met ≤ 4.0
    - 0.0 ≤ clo ≤ 1.5

    Examples
    --------
    >>> import numpy as np
    >>> res = evaluate_vertical_gradient(
    ...     tdb=np.array([24.0, 25.0]),
    ...     tr=np.array([24.0, 25.0]),
    ...     vr=np.array([0.1, 0.1]),
    ...     rh=np.array([50.0, 50.0]),
    ...     met=1.2, clo=0.5,
    ...     vertical_tmp_grad=np.array([2.0, 7.0]),
    ... )
    >>> res.ppd_vg.shape
    (2,)
    >>> bool(res.acceptability[0])
    True
    """
    from pythermalcomfort.models import vertical_tmp_grad_ppd

    tdb_arr = validate_input_array(tdb, "air_temp_c")
    n = tdb_arr.shape[0]
    tr_arr = validate_input_array(tr, "radiant_temp_c")
    vr_arr = validate_input_array(vr, "air_velocity_ms")
    rh_arr = validate_input_array(rh, "relative_humidity_pct")
    met_arr = np.full(n, met, dtype=float) if np.isscalar(met) else np.asarray(met, dtype=float)
    clo_arr = np.full(n, clo, dtype=float) if np.isscalar(clo) else np.asarray(clo, dtype=float)
    vtg_arr = (
        np.full(n, vertical_tmp_grad, dtype=float)
        if np.isscalar(vertical_tmp_grad)
        else np.asarray(vertical_tmp_grad, dtype=float)
    )

    ppd_vals = np.empty(n, dtype=float)
    accept_vals = np.empty(n, dtype=bool)

    for i in range(n):
        result = vertical_tmp_grad_ppd(
            tdb=float(tdb_arr[i]),
            tr=float(tr_arr[i]),
            vr=float(vr_arr[i]),
            rh=float(rh_arr[i]),
            met=float(met_arr[i]),
            clo=float(clo_arr[i]),
            vertical_tmp_grad=float(vtg_arr[i]),
        )
        ppd_vals[i] = result.ppd_vg
        accept_vals[i] = result.acceptability

    logger.debug("vertical_gradient: %d samples, mean PPD=%.1f%%", n, float(np.nanmean(ppd_vals)))

    return VerticalGradientResult(
        ppd_vg=ppd_vals,
        acceptability=accept_vals,
        vertical_tmp_grad=vtg_arr,
    )


def local_discomfort_score(
    ppd_ad: np.ndarray | None = None,
    ppd_vg: np.ndarray | None = None,
) -> np.ndarray:
    """Convert local discomfort PPD values to a 0-100 score.

    Combines ankle draft and vertical gradient PPD into a single score.
    Higher is better (100 = no local discomfort).

    Parameters
    ----------
    ppd_ad : np.ndarray or None
        PPD due to ankle draft (%). If None, this component is skipped.
    ppd_vg : np.ndarray or None
        PPD due to vertical gradient (%). If None, this component is skipped.

    Returns
    -------
    np.ndarray
        Local discomfort score (0-100).

    Notes
    -----
    The score is the mean of ``100 - PPD`` across all provided components:

    .. math::

        \\text{score} = \\frac{1}{k} \\sum_{i=1}^{k} \\text{clip}(100 - \\text{PPD}_i, 0, 100)

    where *k* is the number of components provided (1 or 2).

    Examples
    --------
    >>> import numpy as np
    >>> scores = local_discomfort_score(
    ...     ppd_ad=np.array([15.0, 25.0]),
    ...     ppd_vg=np.array([5.0, 30.0]),
    ... )
    >>> scores.shape
    (2,)
    >>> round(float(scores[0]), 1)
    90.0
    """
    components: list[np.ndarray] = []
    if ppd_ad is not None:
        components.append(np.clip(100.0 - np.asarray(ppd_ad, dtype=float), 0.0, 100.0))
    if ppd_vg is not None:
        components.append(np.clip(100.0 - np.asarray(ppd_vg, dtype=float), 0.0, 100.0))
    if not components:
        raise ValueError("At least one of ppd_ad or ppd_vg must be provided.")
    return np.mean(components, axis=0)
