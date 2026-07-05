"""Tests for input validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.core.exceptions import MissingSensorDataError, OutOfRangeError
from comfio.utils.validation import (
    PHYSICAL_BOUNDS,
    check_required_columns,
    validate_array,
    validate_input_array,
    validate_sensor_column,
    warn_if_sparse,
)


class TestValidateArray:
    def test_clean_data_passes(self) -> None:
        arr = np.array([20.0, 22.0, 24.0, 26.0])
        result = validate_array(arr, bounds=(-40.0, 60.0), label="air_temp")
        assert result.n_dropped == 0
        assert result.n_out_of_range == 0
        np.testing.assert_array_equal(result.data, arr)

    def test_nan_interpolate(self) -> None:
        arr = np.array([20.0, np.nan, 24.0, 26.0])
        result = validate_array(arr, bounds=(-40.0, 60.0), nan_strategy="interpolate", label="test")
        assert not np.any(np.isnan(result.data))
        assert len(result.warnings) > 0

    def test_nan_drop(self) -> None:
        arr = np.array([20.0, np.nan, 24.0, 26.0])
        result = validate_array(arr, bounds=(-40.0, 60.0), nan_strategy="drop", label="test")
        assert result.n_dropped == 1
        assert len(result.data) == 3

    def test_nan_fill_zero(self) -> None:
        arr = np.array([20.0, np.nan, 24.0])
        result = validate_array(arr, bounds=(-40.0, 60.0), nan_strategy="fill_zero", label="test")
        assert result.data[1] == 0.0

    def test_nan_raise(self) -> None:
        arr = np.array([20.0, np.nan, 24.0])
        with pytest.raises(OutOfRangeError):
            validate_array(arr, bounds=(-40.0, 60.0), nan_strategy="raise", label="test")

    def test_out_of_range_clipped(self) -> None:
        arr = np.array([20.0, 70.0, 24.0])  # 70 is slightly above 60
        result = validate_array(arr, bounds=(-40.0, 60.0), label="test")
        assert result.n_out_of_range == 1
        assert result.data[1] == 60.0  # clipped

    def test_wildly_out_of_range_raises(self) -> None:
        arr = np.array([20.0, 200.0, 24.0])  # 200 >> 60 + 50
        with pytest.raises(OutOfRangeError):
            validate_array(arr, bounds=(-40.0, 60.0), label="test")


class TestValidateSensorColumn:
    def test_known_column(self) -> None:
        arr = np.array([22.0, 24.0, 26.0])
        result = validate_sensor_column(arr, "air_temp_c")
        assert result.n_out_of_range == 0

    def test_unknown_column_raises(self) -> None:
        arr = np.array([22.0])
        with pytest.raises(KeyError):
            validate_sensor_column(arr, "unknown_column")


class TestCheckRequiredColumns:
    def test_all_present(self) -> None:
        check_required_columns(["a", "b", "c"], ["a", "b"])

    def test_missing_raises(self) -> None:
        with pytest.raises(MissingSensorDataError):
            check_required_columns(["a", "b"], ["a", "c"])


class TestWarnIfSparse:
    def test_sparse_warning(self) -> None:
        arr = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        with pytest.warns(UserWarning):
            warn_if_sparse(arr, "test_sensor", threshold=0.3)

    def test_no_warning_when_clean(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        warn_if_sparse(arr, "test_sensor", threshold=0.3)


class TestValidateInputArray:
    def test_returns_cleaned_array(self) -> None:
        arr = np.array([22.0, 24.0, 26.0])
        result = validate_input_array(arr, "air_temp_c")
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, arr)

    def test_unknown_key_returns_as_is(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = validate_input_array(arr, "nonexistent_key")
        np.testing.assert_array_equal(result, arr)

    def test_flattens_multidimensional(self) -> None:
        arr = np.array([[22.0, 24.0], [26.0, 28.0]])
        result = validate_input_array(arr, "air_temp_c")
        assert result.ndim == 1
        assert len(result) == 4

    def test_clips_out_of_range(self) -> None:
        arr = np.array([22.0, 100.0, 24.0])  # 100 > 60 max
        result = validate_input_array(arr, "air_temp_c")
        assert result[1] == 60.0  # clipped

    def test_nan_interpolation(self) -> None:
        arr = np.array([22.0, np.nan, 24.0])
        result = validate_input_array(arr, "air_temp_c")
        assert not np.any(np.isnan(result))


class TestPhysicalBoundsEntries:
    def test_pollutant_bounds_exist(self) -> None:
        for key in ["pm25_ugm3", "pm10_ugm3", "tvoc_ugm3", "formaldehyde_ppb", "co_ppm"]:
            assert key in PHYSICAL_BOUNDS

    def test_outdoor_temp_bounds_exist(self) -> None:
        for key in ["outdoor_temp_c", "prevailing_mean_outdoor_c", "running_mean_outdoor_c"]:
            assert key in PHYSICAL_BOUNDS

    def test_advanced_bounds_exist(self) -> None:
        for key in ["spectral_power", "impulse_response", "room_volume_m3", "n_occupants", "atmospheric_pressure_pa"]:
            assert key in PHYSICAL_BOUNDS
