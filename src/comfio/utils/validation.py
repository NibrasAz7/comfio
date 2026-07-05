"""Pre-calculation input validation and boundary checks.

All incoming DataFrames/arrays are routed through this module first.
Clean the data, drop or interpolate NaNs, ensure variables are within
realistic physical bounds, and then pass the clean data to the physics
modules.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from comfio.core.exceptions import OutOfRangeError

# Physical bounds for common IEQ variables.
# Keys map to expected sensor column names. Values are (min, max) tuples.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    # Core IEQ variables
    "air_temp_c": (-40.0, 60.0),
    "radiant_temp_c": (-40.0, 100.0),
    "relative_humidity_pct": (0.0, 100.0),
    "air_velocity_ms": (0.0, 50.0),
    "illuminance_lux": (0.0, 100_000.0),
    "co2_ppm": (0.0, 10_000.0),
    "noise_laeq_db": (0.0, 200.0),
    "metabolic_rate_met": (0.0, 10.0),
    "clothing_insulation_clo": (0.0, 4.0),
    # Advanced domain sensor columns (previously missing from PHYSICAL_BOUNDS)
    "spectral_power": (0.0, 1e6),
    "impulse_response": (-1.0, 1.0),
    "room_volume_m3": (0.0, 1e6),
    "n_occupants": (0.0, 1_000.0),
    "atmospheric_pressure_pa": (80_000.0, 120_000.0),
    # IAQ pollutant columns
    "pm25_ugm3": (0.0, 1_000.0),
    "pm10_ugm3": (0.0, 2_000.0),
    "tvoc_ugm3": (0.0, 10_000.0),
    "formaldehyde_ppb": (0.0, 5_000.0),
    "co_ppm": (0.0, 1_000.0),
    # Outdoor temperature columns (for adaptive models)
    "outdoor_temp_c": (-60.0, 60.0),
    "prevailing_mean_outdoor_c": (-60.0, 60.0),
    "running_mean_outdoor_c": (-60.0, 60.0),
}

NaN_STRATEGY = Literal["drop", "interpolate", "fill_zero", "raise"]


@dataclass
class ValidationResult:
    """Result of validating a sensor data array.

    Attributes
    ----------
    data : np.ndarray
        The cleaned data array.
    n_dropped : int
        Number of rows/elements dropped due to NaN.
    n_out_of_range : int
        Number of values that were clipped to physical bounds.
    warnings : list[str]
        Human-readable warning messages generated during validation.
    """

    data: np.ndarray
    n_dropped: int = 0
    n_out_of_range: int = 0
    warnings: list[str] = field(default_factory=list)


def validate_array(
    data: np.ndarray,
    bounds: tuple[float, float],
    nan_strategy: NaN_STRATEGY = "interpolate",
    label: str = "unknown",
) -> ValidationResult:
    """Validate a 1-D sensor array against physical bounds.

    Parameters
    ----------
    data : np.ndarray
        1-D array of sensor readings.
    bounds : tuple[float, float]
        (min, max) physically realistic range.
    nan_strategy : {"drop", "interpolate", "fill_zero", "raise"}
        How to handle NaN values.
    label : str
        Human-readable name for the variable (used in warnings).

    Returns
    -------
    ValidationResult
        Cleaned data and metadata about what was changed.

    Raises
    ------
    OutOfRangeError
        If values exceed bounds by more than 50% of the range (likely a
        sensor malfunction, not a measurement at the edge of validity).

    Notes
    -----
    NaN handling strategies:

    - **drop**: Remove NaN elements (shortens array)
    - **interpolate**: Linear interpolation between nearest valid values
    - **fill_zero**: Replace NaN with 0.0
    - **raise**: Raise ``OutOfRangeError``

    Values outside physical bounds but within 50% of the range are
    silently clipped.  Values exceeding bounds by >50% raise an error
    (likely sensor malfunction).

    Examples
    --------
    >>> import numpy as np
    >>> result = validate_array(
    ...     np.array([22.0, 25.0, np.nan, 30.0]),
    ...     bounds=(-40.0, 60.0),
    ...     label="air_temp",
    ... )
    >>> result.data.shape
    (4,)
    >>> result.n_out_of_range
    0
    """
    arr = np.asarray(data, dtype=float).copy()
    lo, hi = bounds
    wallys: list[str] = []

    # --- NaN handling ---
    nan_mask = np.isnan(arr)
    n_nan = int(np.sum(nan_mask))

    if n_nan > 0:
        if nan_strategy == "raise":
            raise OutOfRangeError(f"{label}: {n_nan} NaN values found and strategy is 'raise'.")
        elif nan_strategy == "drop":
            arr = arr[~nan_mask]
            wallys.append(f"{label}: dropped {n_nan} NaN values.")
        elif nan_strategy == "fill_zero":
            arr[nan_mask] = 0.0
            wallys.append(f"{label}: filled {n_nan} NaN values with 0.")
        elif nan_strategy == "interpolate":
            if arr.size > 1:
                valid_idx = np.where(~nan_mask)[0]
                if valid_idx.size > 0:
                    arr[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, arr[valid_idx])
                else:
                    arr[nan_mask] = lo
            else:
                arr[nan_mask] = lo
            wallys.append(f"{label}: interpolated {n_nan} NaN values.")

    # --- Bounds checking ---
    out_mask = (arr < lo) | (arr > hi)
    n_out = int(np.sum(out_mask))

    # Hard fail if values are wildly out of range (>50% beyond bounds)
    range_span = hi - lo
    wild_mask = (arr < lo - 0.5 * range_span) | (arr > hi + 0.5 * range_span)
    if np.any(wild_mask):
        raise OutOfRangeError(
            f"{label}: {int(np.sum(wild_mask))} values exceed physical bounds "
            f"({lo}, {hi}) by more than 50% of range. Likely sensor malfunction."
        )

    if n_out > 0:
        arr = np.clip(arr, lo, hi)
        wallys.append(f"{label}: clipped {n_out} values to bounds ({lo}, {hi}).")

    n_dropped = n_nan if nan_strategy == "drop" else 0

    return ValidationResult(
        data=arr,
        n_dropped=n_dropped,
        n_out_of_range=n_out,
        warnings=wallys,
    )


def validate_sensor_column(
    data: np.ndarray,
    column_name: str,
    nan_strategy: NaN_STRATEGY = "interpolate",
) -> ValidationResult:
    """Validate a sensor column using the predefined physical bounds.

    Parameters
    ----------
    data : np.ndarray
        1-D array of sensor readings.
    column_name : str
        Name matching a key in ``PHYSICAL_BOUNDS``.
    nan_strategy : {"drop", "interpolate", "fill_zero", "raise"}
        NaN handling strategy.

    Returns
    -------
    ValidationResult
        Cleaned data and metadata.

    Raises
    ------
    KeyError
        If ``column_name`` is not in ``PHYSICAL_BOUNDS``.
    """
    if column_name not in PHYSICAL_BOUNDS:
        raise KeyError(
            f"No physical bounds defined for '{column_name}'. "
            f"Available: {list(PHYSICAL_BOUNDS.keys())}"
        )
    return validate_array(data, PHYSICAL_BOUNDS[column_name], nan_strategy, label=column_name)


def check_required_columns(
    available: list[str],
    required: list[str],
) -> None:
    """Verify that all required sensor columns are present.

    Parameters
    ----------
    available : list[str]
        Column names present in the data.
    required : list[str]
        Column names that must be present.

    Raises
    ------
    MissingSensorDataError
        If any required column is missing.
    """
    from comfio.core.exceptions import MissingSensorDataError

    missing = [col for col in required if col not in available]
    if missing:
        raise MissingSensorDataError(
            f"Missing required sensor columns: {missing}. Available columns: {available}"
        )


def warn_if_sparse(data: np.ndarray, label: str, threshold: float = 0.3) -> None:
    """Emit a warning if the fraction of NaN values exceeds a threshold.

    Parameters
    ----------
    data : np.ndarray
        1-D array of sensor readings.
    label : str
        Variable name for the warning message.
    threshold : float, default 0.3
        Maximum acceptable fraction of NaN values (0-1).
    """
    frac_nan = np.mean(np.isnan(data))
    if frac_nan > threshold:
        warnings.warn(
            f"{label}: {frac_nan:.1%} of values are NaN (threshold: {threshold:.0%}). "
            f"Results for this variable may be unreliable.",
            UserWarning,
            stacklevel=2,
        )


def validate_input_array(
    data: np.ndarray,
    bounds_key: str,
    nan_strategy: NaN_STRATEGY = "interpolate",
) -> np.ndarray:
    """Validate an input array using PHYSICAL_BOUNDS and return cleaned data.

    Convenience wrapper around ``validate_sensor_column`` that returns
    only the cleaned array (not the full ``ValidationResult``).
    Intended for use inside ``evaluate_*()`` domain functions.

    If ``bounds_key`` is not in ``PHYSICAL_BOUNDS``, the data is returned
    as-is (converted to float) without validation.

    Parameters
    ----------
    data : np.ndarray
        Raw input array.
    bounds_key : str
        Key in ``PHYSICAL_BOUNDS`` matching this variable.
    nan_strategy : {"drop", "interpolate", "fill_zero", "raise"}
        NaN handling strategy.

    Returns
    -------
    np.ndarray
        Cleaned 1-D float array.

    Examples
    --------
    >>> import numpy as np
    >>> cleaned = validate_input_array(
    ...     np.array([22.0, 25.0, 30.0]),
    ...     "air_temp_c",
    ... )
    >>> cleaned.shape
    (3,)
    >>> float(cleaned[0])
    22.0
    """
    arr = np.asarray(data, dtype=float).flatten()
    if bounds_key not in PHYSICAL_BOUNDS:
        return arr
    return validate_array(arr, PHYSICAL_BOUNDS[bounds_key], nan_strategy, label=bounds_key).data
