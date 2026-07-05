"""Tests for SensorData data handler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from comfio.core.data_handler import SensorData
from comfio.core.exceptions import MissingSensorDataError


class TestSensorData:
    def test_auto_detect_columns(self, mock_sensor_df: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df)
        assert "air_temp_c" in sensor.column_map
        assert "illuminance_lux" in sensor.column_map
        assert "co2_ppm" in sensor.column_map
        assert "noise_laeq_db" in sensor.column_map

    def test_auto_detect_aliases(self, mock_sensor_df_partial: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df_partial)
        # Should detect tdb → air_temp_c, co2 → co2_ppm, etc.
        assert "air_temp_c" in sensor.column_map
        assert "co2_ppm" in sensor.column_map

    def test_available_domains_full(self, mock_sensor_df: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df)
        domains = sensor.available_domains()
        assert "thermal" in domains
        assert "visual" in domains
        assert "acoustic" in domains
        assert "iaq" in domains

    def test_available_domains_partial(self, mock_sensor_df_partial: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df_partial)
        domains = sensor.available_domains()
        assert "thermal" in domains
        assert "iaq" in domains
        assert "visual" not in domains
        assert "acoustic" not in domains

    def test_get_column(self, mock_sensor_df: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df)
        arr = sensor.get_column("air_temp_c")
        assert arr.shape == (100,)
        assert arr.dtype == float

    def test_get_column_missing_raises(self, mock_sensor_df: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df)
        with pytest.raises(MissingSensorDataError):
            sensor.get_column("nonexistent_column")

    def test_validate(self, mock_sensor_df_with_nan: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df_with_nan)
        assert not sensor.validated
        sensor.validate()
        assert sensor.validated
        # After validation, NaN should be gone
        arr = sensor.get_validated("air_temp_c")
        assert not np.any(np.isnan(arr))

    def test_get_validated_caches(self, mock_sensor_df: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df)
        arr1 = sensor.get_validated("air_temp_c")
        assert "air_temp_c" in sensor.validation_results
        arr2 = sensor.get_validated("air_temp_c")
        np.testing.assert_array_equal(arr1, arr2)

    def test_len(self, mock_sensor_df: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df)
        assert len(sensor) == 100

    def test_repr(self, mock_sensor_df: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df)
        repr_str = repr(sensor)
        assert "SensorData" in repr_str
        assert "n_rows=100" in repr_str

    def test_custom_column_map(self, mock_sensor_df: pd.DataFrame) -> None:
        # Rename columns and provide explicit map
        df = mock_sensor_df.rename(columns={"air_temp_c": "temp_sensor_1"})
        sensor = SensorData(df=df, column_map={"air_temp_c": "temp_sensor_1"})
        arr = sensor.get_column("air_temp_c")
        assert arr.shape == (100,)

    def test_custom_column_map_invalid_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(MissingSensorDataError):
            SensorData(df=df, column_map={"air_temp_c": "nonexistent"})

    def test_get_timestamps(self, mock_sensor_df: pd.DataFrame) -> None:
        sensor = SensorData(df=mock_sensor_df, timestamp_col="timestamp")
        ts = sensor.get_timestamps()
        assert ts is not None
        assert len(ts) == 100

    def test_get_timestamps_none(self) -> None:
        df = pd.DataFrame({"air_temp_c": [22.0, 24.0]})
        sensor = SensorData(df=df)
        assert sensor.get_timestamps() is None
