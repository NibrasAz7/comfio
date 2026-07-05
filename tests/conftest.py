"""Shared pytest fixtures for comfio tests.

Provides mock sensor DataFrames and arrays for all domains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def n_samples() -> int:
    """Number of samples in mock time-series data."""
    return 100


@pytest.fixture
def mock_thermal_arrays(n_samples: int) -> dict[str, np.ndarray]:
    """Mock thermal comfort sensor arrays (realistic office conditions)."""
    rng = np.random.default_rng(42)
    return {
        "tdb": rng.normal(24.0, 1.5, n_samples),  # 24°C ± 1.5
        "tr": rng.normal(24.0, 1.0, n_samples),  # radiant ~ air temp
        "vr": rng.normal(0.1, 0.03, n_samples),  # 0.1 m/s
        "rh": rng.normal(50.0, 5.0, n_samples),  # 50% RH
        "met": np.full(n_samples, 1.2),  # sedentary
        "clo": np.full(n_samples, 0.5),  # light clothing
    }


@pytest.fixture
def mock_visual_array(n_samples: int) -> np.ndarray:
    """Mock illuminance array (office conditions, ~500 lux)."""
    rng = np.random.default_rng(42)
    return rng.normal(500.0, 50.0, n_samples)


@pytest.fixture
def mock_acoustic_array(n_samples: int) -> np.ndarray:
    """Mock L_Aeq noise array (office conditions, ~40 dB)."""
    rng = np.random.default_rng(42)
    return rng.normal(40.0, 5.0, n_samples)


@pytest.fixture
def mock_iaq_array(n_samples: int) -> np.ndarray:
    """Mock CO₂ array (office conditions, ~800 ppm)."""
    rng = np.random.default_rng(42)
    return rng.normal(800.0, 100.0, n_samples)


@pytest.fixture
def mock_sensor_df(
    mock_thermal_arrays: dict[str, np.ndarray],
    mock_visual_array: np.ndarray,
    mock_acoustic_array: np.ndarray,
    mock_iaq_array: np.ndarray,
    n_samples: int,
) -> pd.DataFrame:
    """Full mock sensor DataFrame with all four domains."""
    dates = pd.date_range("2025-01-01", periods=n_samples, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "air_temp_c": mock_thermal_arrays["tdb"],
            "radiant_temp_c": mock_thermal_arrays["tr"],
            "air_velocity_ms": mock_thermal_arrays["vr"],
            "relative_humidity_pct": mock_thermal_arrays["rh"],
            "metabolic_rate_met": mock_thermal_arrays["met"],
            "clothing_insulation_clo": mock_thermal_arrays["clo"],
            "illuminance_lux": mock_visual_array,
            "noise_laeq_db": mock_acoustic_array,
            "co2_ppm": mock_iaq_array,
        }
    )


@pytest.fixture
def mock_sensor_df_partial(
    mock_thermal_arrays: dict[str, np.ndarray],
    mock_iaq_array: np.ndarray,
    n_samples: int,
) -> pd.DataFrame:
    """Partial mock sensor DataFrame (thermal + IAQ only, no visual/acoustic)."""
    dates = pd.date_range("2025-01-01", periods=n_samples, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "tdb": mock_thermal_arrays["tdb"],
            "tr": mock_thermal_arrays["tr"],
            "vr": mock_thermal_arrays["vr"],
            "rh": mock_thermal_arrays["rh"],
            "met": mock_thermal_arrays["met"],
            "clo": mock_thermal_arrays["clo"],
            "co2": mock_iaq_array,
        }
    )


@pytest.fixture
def mock_sensor_df_with_nan(
    mock_thermal_arrays: dict[str, np.ndarray],
    mock_visual_array: np.ndarray,
    mock_acoustic_array: np.ndarray,
    mock_iaq_array: np.ndarray,
    n_samples: int,
) -> pd.DataFrame:
    """Mock sensor DataFrame with some NaN values injected."""
    df = pd.DataFrame(
        {
            "air_temp_c": mock_thermal_arrays["tdb"].copy(),
            "radiant_temp_c": mock_thermal_arrays["tr"].copy(),
            "air_velocity_ms": mock_thermal_arrays["vr"].copy(),
            "relative_humidity_pct": mock_thermal_arrays["rh"].copy(),
            "metabolic_rate_met": mock_thermal_arrays["met"],
            "clothing_insulation_clo": mock_thermal_arrays["clo"],
            "illuminance_lux": mock_visual_array.copy(),
            "noise_laeq_db": mock_acoustic_array.copy(),
            "co2_ppm": mock_iaq_array.copy(),
        }
    )
    # Inject NaNs in 10% of rows for air_temp_c
    rng = np.random.default_rng(99)
    nan_idx = rng.choice(n_samples, size=10, replace=False)
    df.loc[nan_idx, "air_temp_c"] = np.nan
    return df


@pytest.fixture
def mock_pollutant_arrays(n_samples: int) -> dict[str, np.ndarray]:
    """Mock IAQ pollutant arrays (typical office conditions)."""
    rng = np.random.default_rng(42)
    return {
        "pm25": rng.normal(8.0, 2.0, n_samples),  # ~8 µg/m³
        "pm10": rng.normal(15.0, 3.0, n_samples),  # ~15 µg/m³
        "tvoc": rng.normal(150.0, 30.0, n_samples),  # ~150 µg/m³
        "formaldehyde": rng.normal(20.0, 5.0, n_samples),  # ~20 ppb
        "co": rng.normal(1.5, 0.5, n_samples),  # ~1.5 ppm
    }


@pytest.fixture
def mock_outdoor_temp(n_samples: int) -> np.ndarray:
    """Mock outdoor temperature array for adaptive comfort."""
    rng = np.random.default_rng(42)
    return rng.normal(20.0, 5.0, n_samples)


@pytest.fixture
def mock_tsv_votes() -> np.ndarray:
    """Mock sparse TSV votes (occupant feedback, -3 to +3)."""
    return np.array([-2, -1, 0, 0, 1, 1, 2, -1, 0, 1], dtype=float)


@pytest.fixture
def mock_tsv_timestamps() -> np.ndarray:
    """Mock timestamps for sparse TSV votes (hourly, 10 samples)."""
    return np.arange(10, dtype=float)


@pytest.fixture
def mock_target_timestamps(n_samples: int) -> np.ndarray:
    """Mock dense target timestamps for TSV augmentation."""
    return np.arange(n_samples, dtype=float)


@pytest.fixture
def mock_pmv_tsv_pairs(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Mock paired PMV/TSV data for personalisation training."""
    rng = np.random.default_rng(42)
    pmv = rng.normal(0.3, 0.5, n_samples)
    # TSV = 1.1 * PMV + 0.2 + noise (so regression can recover approx alpha=1.1, beta=0.2)
    tsv = 1.1 * pmv + 0.2 + rng.normal(0, 0.3, n_samples)
    tsv = np.clip(np.round(tsv), -3, 3)
    return pmv, tsv


@pytest.fixture
def mock_seasonal_dates(n_samples: int) -> list:
    """Mock dates spanning multiple seasons for seasonal personalisation."""
    from datetime import date, timedelta

    return [date(2025, 1, 15) + timedelta(days=int(i * 3.65)) for i in range(n_samples)]
