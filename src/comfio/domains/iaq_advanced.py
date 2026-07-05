"""Advanced IAQ module — ventilation rate calculation and psychrometrics.

Provides CO₂-based Air Change Rate (ACH) estimation and full
psychrometric property calculations. These functions require an optional
extra:

    pip install comfio[psychrometrics]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from comfio.domains.iaq import CO2_OUTDOOR_BASELINE


@dataclass
class VentilationResult:
    """Result of a ventilation adequacy evaluation.

    Attributes
    ----------
    ach : float
        Estimated Air Change Rate (air changes per hour).
    ach_method : str
        Method used: "co2_decay" or "steady_state".
    ventilation_efficiency : float
        Ratio of outdoor air supply to total supply air (0-1).
    co2_peak : float
        Peak CO₂ concentration observed (ppm).
    co2_steady_state : float
        Estimated steady-state CO₂ concentration (ppm).
    compliant : bool
        True if ACH meets the minimum for the occupancy type.
    score : float
        Ventilation comfort score (0-100), higher is better.
    occupancy_type : str
        Occupancy type used for compliance check.
    """

    ach: float
    ach_method: str
    ventilation_efficiency: float
    co2_peak: float
    co2_steady_state: float
    compliant: bool
    score: float
    occupancy_type: str


@dataclass
class PsychrometricResult:
    """Full psychrometric properties of moist air.

    Attributes
    ----------
    tdb : float
        Dry-bulb temperature (°C).
    rh : float
        Relative humidity (0-1).
    pressure : float
        Atmospheric pressure (Pa).
    twb : float
        Wet-bulb temperature (°C).
    tdew : float
        Dew-point temperature (°C).
    enthalpy : float
        Moist air enthalpy (J/kg).
    hum_ratio : float
        Humidity ratio (kg water / kg dry air).
    vapor_pressure : float
        Partial pressure of water vapor (Pa).
    moist_air_volume : float
        Specific volume of moist air (m³/kg dry air).
    degree_of_saturation : float
        Degree of saturation (0-1).
    """

    tdb: float
    rh: float
    pressure: float
    twb: float
    tdew: float
    enthalpy: float
    hum_ratio: float
    vapor_pressure: float
    moist_air_volume: float
    degree_of_saturation: float


# Minimum ACH targets per occupancy type (from ASHRAE 62.1, simplified)
MIN_ACH_TARGETS: dict[str, float] = {
    "office": 2.0,
    "classroom": 3.0,
    "meeting_room": 2.5,
    "residential": 0.5,
    "hospital": 4.0,
    "restaurant": 5.0,
    "general": 2.0,
}


def _require_psychrolib() -> Any:
    """Import psychrolib or raise a helpful error.

    Works around a psychrolib 2.5.0 + numba incompatibility where
    ``njit`` returns a plain function (not a Dispatcher) and ``vectorize``
    creates DUFuncs that fail to compile with ``TypingError``.
    The fix restores the pure Python originals from ``func_list``.
    """
    try:
        import psychrolib  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "psychrolib is required for psychrometric calculations. "
            "Install it with: pip install comfio[psychrometrics]"
        ) from None

    if not getattr(psychrolib, "_comfio_patched", False):
        # Restore pure Python functions, bypassing numba vectorize
        for func_name, func_src in psychrolib.func_list:
            psychrolib.__dict__[func_name] = func_src
        psychrolib.has_numba = False
        # Patch SetUnitSystem to skip njit recompilation
        _orig_set_unit = psychrolib.SetUnitSystem

        def _safe_set_unit_system(units: Any) -> None:
            psychrolib.PSYCHROLIB_UNITS = units
            psychrolib.PSYCHROLIB_TOLERANCE = 0.001 * 9.0 / 5.0 if units == psychrolib.IP else 0.001

        psychrolib.SetUnitSystem = _safe_set_unit_system
        psychrolib._comfio_patched = True

    return psychrolib


def evaluate_ventilation(
    co2: np.ndarray,
    timestamps: np.ndarray | None = None,
    outdoor_co2: float = CO2_OUTDOOR_BASELINE,
    occupancy_type: str = "general",
    room_volume: float | None = None,
    n_occupants: int | None = None,
    co2_generation_rate: float = 0.005,
) -> VentilationResult:
    """Estimate ventilation rate (ACH) from CO₂ concentration data.

    Uses the CO₂ decay method when a decay phase is identifiable in the
    data, or the steady-state method when occupancy and room parameters
    are provided.

    Parameters
    ----------
    co2 : np.ndarray
        CO₂ concentration time series in ppm.
    timestamps : np.ndarray or None
        Timestamps in seconds. If None, assumes 1-hour intervals.
    outdoor_co2 : float
        Outdoor CO₂ baseline in ppm (default 420).
    occupancy_type : str
        Occupancy type key from ``MIN_ACH_TARGETS`` for compliance.
    room_volume : float or None
        Room volume in m³ (required for steady-state method).
    n_occupants : int or None
        Number of occupants (required for steady-state method).
    co2_generation_rate : float
        CO₂ generation rate per person in L/s (default 0.005).

    Returns
    -------
    VentilationResult
        ACH, ventilation efficiency, compliance, and score.

    Notes
    -----
    **CO₂ decay method**: When occupants leave, CO₂ decays exponentially.
    The decay rate is related to ACH by: ACH = -slope * 3600 (if timestamps
    are in seconds). This method requires a clear decay phase in the data.

    **Steady-state method**: Uses the steady-state CO₂ balance:
    C_ss = C_out + G * N / (Q * V)
    where G is per-person generation rate, N is occupants, Q is ACH,
    and V is room volume.
    """
    co2_arr = np.asarray(co2, dtype=float).flatten()

    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=float).flatten()
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 3600.0
    else:
        dt = 3600.0  # 1 hour default
        ts = np.arange(len(co2_arr), dtype=float) * dt

    co2_peak = float(np.max(co2_arr))

    # Try CO₂ decay method first
    # Look for a monotonic decay phase (occupants left)
    ach_value = 0.0
    ach_method = "co2_decay"

    # Find the decay phase: longest run of consecutive decreases
    diffs = np.diff(co2_arr)
    decay_mask = diffs < 0
    if np.any(decay_mask):
        # Find longest consecutive decay
        best_start, best_len = 0, 0
        cur_start, cur_len = 0, 0
        for pancake in range(len(decay_mask)):
            if decay_mask[pancake]:
                if cur_len == 0:
                    cur_start = pancake
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                cur_len = 0
        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len

        if best_len >= 3:  # Need at least 3 points for a fit
            decay_ts = ts[best_start : best_start + best_len + 1]
            decay_co2 = co2_arr[best_start : best_start + best_len + 1]

            # Exponential decay: C(t) = C_out + (C0 - C_out) * exp(-k*t)
            # ln(C(t) - C_out) = ln(C0 - C_out) - k*t
            excess = decay_co2 - outdoor_co2
            excess = np.maximum(excess, 1.0)  # Avoid log(0)

            log_excess = np.log(excess)
            # Linear fit: log_excess = a + b * t
            coeffs = np.polyfit(decay_ts, log_excess, 1)
            decay_rate = -coeffs[0]  # per second

            if decay_rate > 0:
                ach_value = decay_rate * 3600.0  # Convert to per hour

    # If decay method failed, try steady-state
    if ach_value <= 0 and room_volume is not None and n_occupants is not None and n_occupants > 0:
        ach_method = "steady_state"
        # Use the last 20% of data as potential steady state
        n_steady = max(3, len(co2_arr) // 5)
        co2_ss = float(np.mean(co2_arr[-n_steady:]))

        # C_ss = C_out + (G * N) / (ACH * V)
        # ACH = (G * N) / ((C_ss - C_out) * V)
        # G in L/s, convert to m³/h: G * 3.6
        # C in ppm → convert to fraction: / 1e6
        delta_c = (co2_ss - outdoor_co2) / 1e6  # fraction
        if delta_c > 0:
            g_m3h = co2_generation_rate * 3.6  # L/s → m³/h per person
            ach_value = (g_m3h * n_occupants) / (delta_c * room_volume)

    if ach_value <= 0:
        ach_value = 0.0
        ach_method = "unknown"

    # Ventilation efficiency: ratio of outdoor CO₂ gap to total
    co2_ss_est = float(np.mean(co2_arr[-max(3, len(co2_arr) // 5) :]))
    if co2_ss_est > outdoor_co2:
        vent_eff = float(np.clip(outdoor_co2 / co2_ss_est, 0.0, 1.0))
    else:
        vent_eff = 1.0

    # Compliance
    min_ach = MIN_ACH_TARGETS.get(occupancy_type, MIN_ACH_TARGETS["general"])
    compliant = ach_value >= min_ach

    # Score: 100 at 2× min ACH, 50 at min ACH, 0 at 0 ACH
    score = float(np.clip(50.0 * ach_value / min_ach, 0.0, 100.0)) if min_ach > 0 else 50.0

    return VentilationResult(
        ach=ach_value,
        ach_method=ach_method,
        ventilation_efficiency=vent_eff,
        co2_peak=co2_peak,
        co2_steady_state=co2_ss_est,
        compliant=compliant,
        score=score,
        occupancy_type=occupancy_type,
    )


def get_psychrometrics(
    tdb: float,
    rh: float,
    pressure: float = 101325.0,
) -> PsychrometricResult:
    """Calculate full psychrometric properties of moist air.

    Uses PsychroLib (based on ASHRAE Handbook — Fundamentals, 2017).

    Parameters
    ----------
    tdb : float
        Dry-bulb temperature in °C.
    rh : float
        Relative humidity (0-1, not 0-100%).
    pressure : float
        Atmospheric pressure in Pa (default: 101325 Pa at sea level).

    Returns
    -------
    PsychrometricResult
        All psychrometric properties: wet bulb, dew point, enthalpy,
        humidity ratio, vapor pressure, moist air volume, degree of saturation.

    Raises
    ------
    ImportError
        If psychrolib is not installed.
    """
    psychrolib = _require_psychrolib()

    # Set SI units
    psychrolib.SetUnitSystem(psychrolib.SI)

    # Calculate all properties
    twb = float(psychrolib.GetTWetBulbFromRelHum(tdb, rh, pressure))
    tdew = float(psychrolib.GetTDewPointFromRelHum(tdb, rh))
    hum_ratio = float(psychrolib.GetHumRatioFromRelHum(tdb, rh, pressure))
    vapor_pressure = float(psychrolib.GetVapPresFromRelHum(tdb, rh))
    enthalpy = float(psychrolib.GetMoistAirEnthalpy(tdb, hum_ratio))
    moist_air_volume = float(psychrolib.GetMoistAirVolume(tdb, hum_ratio, pressure))
    degree_of_saturation = float(psychrolib.GetDegreeOfSaturation(tdb, hum_ratio, pressure))

    return PsychrometricResult(
        tdb=tdb,
        rh=rh,
        pressure=pressure,
        twb=twb,
        tdew=tdew,
        enthalpy=enthalpy,
        hum_ratio=hum_ratio,
        vapor_pressure=vapor_pressure,
        moist_air_volume=moist_air_volume,
        degree_of_saturation=degree_of_saturation,
    )
