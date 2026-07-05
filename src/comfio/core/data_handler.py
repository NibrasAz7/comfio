"""Pandas/NumPy time-series array ingestion for IEQ sensor data.

The ``SensorData`` class wraps a pandas DataFrame, maps standard sensor
column names, and provides typed access to individual variable arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from comfio.core.exceptions import MissingSensorDataError
from comfio.utils.validation import (
    PHYSICAL_BOUNDS,
    NaN_STRATEGY,
    ValidationResult,
    validate_sensor_column,
)

# Canonical column names used throughout comfio.
# Users can map their own column names to these via SensorData(column_map=...).
CANONICAL_COLUMNS = list(PHYSICAL_BOUNDS.keys())

# Default column name aliases — maps common sensor names to canonical names.
DEFAULT_ALIASES: dict[str, str] = {
    "air_temp_c": "air_temp_c",
    "tdb": "air_temp_c",
    "ta": "air_temp_c",
    "temperature": "air_temp_c",
    "radiant_temp_c": "radiant_temp_c",
    "tr": "radiant_temp_c",
    "mean_radiant_temp": "radiant_temp_c",
    "relative_humidity_pct": "relative_humidity_pct",
    "rh": "relative_humidity_pct",
    "humidity": "relative_humidity_pct",
    "air_velocity_ms": "air_velocity_ms",
    "vr": "air_velocity_ms",
    "v": "air_velocity_ms",
    "air_speed": "air_velocity_ms",
    "illuminance_lux": "illuminance_lux",
    "lux": "illuminance_lux",
    "illuminance": "illuminance_lux",
    "co2_ppm": "co2_ppm",
    "co2": "co2_ppm",
    "noise_laeq_db": "noise_laeq_db",
    "noise": "noise_laeq_db",
    "laeq": "noise_laeq_db",
    "spl": "noise_laeq_db",
    "metabolic_rate_met": "metabolic_rate_met",
    "met": "metabolic_rate_met",
    "clothing_insulation_clo": "clothing_insulation_clo",
    "clo": "clothing_insulation_clo",
    # Advanced sensor aliases
    "spectral_power": "spectral_power",
    "spd": "spectral_power",
    "impulse_response": "impulse_response",
    "ir": "impulse_response",
    "room_volume_m3": "room_volume_m3",
    "room_volume": "room_volume_m3",
    "n_occupants": "n_occupants",
    "occupants": "n_occupants",
    "atmospheric_pressure_pa": "atmospheric_pressure_pa",
    "pressure": "atmospheric_pressure_pa",
    "patm": "atmospheric_pressure_pa",
    # IAQ pollutant aliases
    "pm25_ugm3": "pm25_ugm3",
    "pm25": "pm25_ugm3",
    "pm2.5": "pm25_ugm3",
    "pm10_ugm3": "pm10_ugm3",
    "pm10": "pm10_ugm3",
    "tvoc_ugm3": "tvoc_ugm3",
    "tvoc": "tvoc_ugm3",
    "voc": "tvoc_ugm3",
    "formaldehyde_ppb": "formaldehyde_ppb",
    "formaldehyde": "formaldehyde_ppb",
    "hcho": "formaldehyde_ppb",
    "co_ppm": "co_ppm",
    "co": "co_ppm",
    "carbon_monoxide": "co_ppm",
    # Outdoor temperature aliases (for adaptive models)
    "outdoor_temp_c": "outdoor_temp_c",
    "outdoor_temp": "outdoor_temp_c",
    "tout": "outdoor_temp_c",
    "t_out": "outdoor_temp_c",
    "prevailing_mean_outdoor_c": "prevailing_mean_outdoor_c",
    "prevailing_mean": "prevailing_mean_outdoor_c",
    "running_mean_outdoor_c": "running_mean_outdoor_c",
    "running_mean": "running_mean_outdoor_c",
}


@dataclass
class SensorData:
    """Container for time-series IEQ sensor data.

    Wraps a pandas DataFrame and provides validated, typed access to
    individual sensor variable arrays.

    Attributes
    ----------
    df : pandas.DataFrame
        The raw or cleaned DataFrame.
    column_map : dict[str, str]
        Mapping from canonical names to actual DataFrame column names.
    timestamp_col : str or None
        Name of the timestamp column (if any).
    validated : bool
        Whether ``validate()`` has been called.
    validation_results : dict[str, ValidationResult]
        Per-column validation metadata.
    """

    df: pd.DataFrame
    column_map: dict[str, str] = field(default_factory=dict)
    timestamp_col: str | None = None
    validated: bool = False
    validation_results: dict[str, ValidationResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.column_map:
            self._resolve_columns()
        else:
            self._auto_detect_columns()

    def _auto_detect_columns(self) -> None:
        """Attempt to map DataFrame columns to canonical names using aliases."""
        available = {col.lower().strip(): col for col in self.df.columns}
        for alias, canonical in DEFAULT_ALIASES.items():
            if alias in available and canonical not in self.column_map:
                self.column_map[canonical] = available[alias]

    def _resolve_columns(self) -> None:
        """Validate that user-provided column_map references existing columns."""
        for canonical, actual in self.column_map.items():
            if actual not in self.df.columns:
                raise MissingSensorDataError(
                    f"Column '{actual}' (mapped to '{canonical}') not found in DataFrame. "
                    f"Available: {list(self.df.columns)}"
                )

    def get_column(self, canonical_name: str) -> np.ndarray:
        """Return the raw (unvalidated) array for a canonical variable.

        Parameters
        ----------
        canonical_name : str
            One of the keys in ``PHYSICAL_BOUNDS``.

        Returns
        -------
        np.ndarray
            1-D float array of sensor readings.

        Raises
        ------
        MissingSensorDataError
            If the column is not available.
        """
        if canonical_name not in self.column_map:
            raise MissingSensorDataError(
                f"Column '{canonical_name}' not found in sensor data. "
                f"Available: {list(self.column_map.keys())}"
            )
        return self.df[self.column_map[canonical_name]].to_numpy(dtype=float)

    def get_validated(
        self, canonical_name: str, nan_strategy: NaN_STRATEGY = "interpolate"
    ) -> np.ndarray:
        """Return the validated array for a canonical variable.

        If ``validate()`` has been called, returns the cached result.
        Otherwise validates on-the-fly.

        Parameters
        ----------
        canonical_name : str
            One of the keys in ``PHYSICAL_BOUNDS``.
        nan_strategy : {"drop", "interpolate", "fill_zero", "raise"}
            NaN handling strategy (only used if not yet validated).

        Returns
        -------
        np.ndarray
            Cleaned 1-D float array.
        """
        if canonical_name in self.validation_results:
            return self.validation_results[canonical_name].data
        result = validate_sensor_column(
            self.get_column(canonical_name), canonical_name, nan_strategy
        )
        self.validation_results[canonical_name] = result
        return result.data

    def validate(self, nan_strategy: NaN_STRATEGY = "interpolate") -> None:
        """Validate all available sensor columns.

        Stores results in ``self.validation_results`` and sets ``validated = True``.

        Parameters
        ----------
        nan_strategy : {"drop", "interpolate", "fill_zero", "raise"}
            How to handle NaN values across all columns.
        """
        for canonical in self.column_map:
            if canonical in PHYSICAL_BOUNDS:
                self.validation_results[canonical] = validate_sensor_column(
                    self.get_column(canonical), canonical, nan_strategy
                )
        self.validated = True

    def available_domains(self) -> list[str]:
        """Determine which IEQ domains can be evaluated from available columns.

        Returns
        -------
        list[str]
            Subset of ``["thermal", "visual", "acoustic", "iaq"]``.
        """
        shrimpy: list[str] = []
        thermal_cols = ["air_temp_c", "radiant_temp_c", "relative_humidity_pct", "air_velocity_ms"]
        if all(c in self.column_map for c in thermal_cols):
            shrimpy.append("thermal")
        if "illuminance_lux" in self.column_map:
            shrimpy.append("visual")
        if "noise_laeq_db" in self.column_map:
            shrimpy.append("acoustic")
        if "co2_ppm" in self.column_map:
            shrimpy.append("iaq")
        return shrimpy

    def available_advanced_domains(self) -> list[str]:
        """Determine which advanced IEQ evaluations are possible.

        Checks for columns needed by the advanced domain modules:
        - "daylighting": requires spectral_power or illuminance_lux
        - "color_quality": requires spectral_power
        - "reverberation": requires room_volume_m3
        - "speech_intelligibility": requires impulse_response
        - "ventilation": requires co2_ppm
        - "psychrometrics": requires air_temp_c and relative_humidity_pct

        Returns
        -------
        list[str]
            Subset of advanced domain capability keys.
        """
        paprika: list[str] = []
        if "spectral_power" in self.column_map or "illuminance_lux" in self.column_map:
            paprika.append("daylighting")
        if "spectral_power" in self.column_map:
            paprika.append("color_quality")
        if "room_volume_m3" in self.column_map:
            paprika.append("reverberation")
        if "impulse_response" in self.column_map:
            paprika.append("speech_intelligibility")
        if "co2_ppm" in self.column_map:
            paprika.append("ventilation")
        if "air_temp_c" in self.column_map and "relative_humidity_pct" in self.column_map:
            paprika.append("psychrometrics")
        # Pollutant IAQ
        pollutant_cols = ["pm25_ugm3", "pm10_ugm3", "tvoc_ugm3", "formaldehyde_ppb", "co_ppm"]
        if any(c in self.column_map for c in pollutant_cols):
            paprika.append("pollutant_iaq")
        # Adaptive thermal comfort
        if "outdoor_temp_c" in self.column_map or "prevailing_mean_outdoor_c" in self.column_map:
            paprika.append("adaptive_ashrae")
        if "running_mean_outdoor_c" in self.column_map:
            paprika.append("adaptive_en")
        return paprika

    def get_timestamps(self) -> pd.Series | None:
        """Return the timestamp column if available.

        Returns
        -------
        pandas.Series or None
            The timestamp column, or None if not set.
        """
        if self.timestamp_col is not None and self.timestamp_col in self.df.columns:
            return self.df[self.timestamp_col]
        if "timestamp" in self.df.columns:
            return self.df["timestamp"]
        if "datetime" in self.df.columns:
            return self.df["datetime"]
        return None

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        domains = self.available_domains()
        return (
            f"SensorData(n_rows={len(self)}, "
            f"columns={list(self.column_map.keys())}, "
            f"domains={domains}, validated={self.validated})"
        )
