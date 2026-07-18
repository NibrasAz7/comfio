"""Tests for weather integration module (mocked — no network calls)."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from comfio.integration import weather


@pytest.fixture
def mock_hourly_df() -> pd.DataFrame:
    """Fake hourly weather data for 7 days."""
    dates = pd.date_range("2025-06-23", "2025-06-30", freq="h")
    rng = np.random.default_rng(42)
    temps = (
        20.0 + 5.0 * np.sin(np.linspace(0, 4 * np.pi, len(dates))) + rng.normal(0, 1, len(dates))
    )
    return pd.DataFrame(
        {
            "temp": temps,
            "rhum": 50.0 + rng.normal(0, 5, len(dates)),
        },
        index=dates,
    )


class TestFetchOutdoorTemperature:
    def test_cache_hit_avoids_network(self, mock_hourly_df: pd.DataFrame, tmp_path) -> None:
        """When cache file exists, no meteostat call should be made."""
        cache_path = tmp_path / "test_cache.parquet"
        mock_hourly_df.to_parquet(cache_path)

        with (
            patch.object(weather, "_cache_key", return_value=cache_path),
            patch("meteostat.Point") as mock_point,
        ):
            df = weather.fetch_outdoor_temperature(
                lat=50.11,
                lon=8.68,
                start=date(2025, 6, 23),
                end=date(2025, 6, 30),
            )
            assert mock_point.call_count == 0  # No network
            assert "temp" in df.columns
            assert len(df) == len(mock_hourly_df)

    def test_network_fetch_and_cache(
        self, mock_hourly_df: pd.DataFrame, tmp_path, monkeypatch
    ) -> None:
        """When no cache, fetch from meteostat and write cache."""
        cache_path = tmp_path / "weather.parquet"
        monkeypatch.setattr(weather, "_cache_key", lambda *a, **k: cache_path)
        monkeypatch.setattr(weather, "_CACHE_DIR", tmp_path)

        mock_meteostat = MagicMock()
        mock_ts = MagicMock()
        mock_ts.fetch.return_value = mock_hourly_df
        mock_meteostat.hourly.return_value = mock_ts

        with patch.dict("sys.modules", {"meteostat": mock_meteostat}):
            df = weather.fetch_outdoor_temperature(
                lat=50.11,
                lon=8.68,
                start=date(2025, 6, 23),
                end=date(2025, 6, 30),
            )

        assert "temp" in df.columns
        assert cache_path.exists()  # Cache was written

    def test_empty_data_raises(self, tmp_path, monkeypatch) -> None:
        """Empty DataFrame from meteostat should raise ValueError."""
        monkeypatch.setattr(weather, "_cache_key", lambda *a, **k: tmp_path / "x.parquet")

        mock_meteostat = MagicMock()
        mock_ts = MagicMock()
        mock_ts.fetch.return_value = pd.DataFrame()
        mock_meteostat.hourly.return_value = mock_ts

        with (
            patch.dict("sys.modules", {"meteostat": mock_meteostat}),
            pytest.raises(ValueError, match="No weather data"),
        ):
            weather.fetch_outdoor_temperature(
                lat=0.0,
                lon=0.0,
                start=date(2025, 1, 1),
                end=date(2025, 1, 2),
            )

    def test_datetime_inputs_accepted(
        self, mock_hourly_df: pd.DataFrame, tmp_path, monkeypatch
    ) -> None:
        """datetime.datetime inputs should be accepted (converted to date)."""
        cache_path = tmp_path / "dt.parquet"
        monkeypatch.setattr(weather, "_cache_key", lambda *a, **k: cache_path)

        mock_meteostat = MagicMock()
        mock_ts = MagicMock()
        mock_ts.fetch.return_value = mock_hourly_df
        mock_meteostat.hourly.return_value = mock_ts

        with patch.dict("sys.modules", {"meteostat": mock_meteostat}):
            df = weather.fetch_outdoor_temperature(
                lat=50.11,
                lon=8.68,
                start=datetime(2025, 6, 23, 12, 0),
                end=datetime(2025, 6, 30, 12, 0),
            )
        assert len(df) > 0


class TestFetchPrevailingTemp:
    def test_returns_scalar_array(
        self, mock_hourly_df: pd.DataFrame, tmp_path, monkeypatch
    ) -> None:
        cache_path = tmp_path / "prevail.parquet"
        monkeypatch.setattr(weather, "_cache_key", lambda *a, **k: cache_path)

        mock_meteostat = MagicMock()
        mock_ts = MagicMock()
        mock_ts.fetch.return_value = mock_hourly_df
        mock_meteostat.hourly.return_value = mock_ts

        with patch.dict("sys.modules", {"meteostat": mock_meteostat}):
            result = weather.fetch_prevailing_temp(
                lat=50.11,
                lon=8.68,
                end_date=date(2025, 6, 30),
                days=7,
            )

        assert isinstance(result, np.ndarray)
        assert result.shape == ()
        assert -20.0 < float(result) < 60.0  # Reasonable temperature


class TestFetchRunningMean:
    def test_returns_scalar_array(
        self, mock_hourly_df: pd.DataFrame, tmp_path, monkeypatch
    ) -> None:
        cache_path = tmp_path / "rm.parquet"
        monkeypatch.setattr(weather, "_cache_key", lambda *a, **k: cache_path)

        mock_meteostat = MagicMock()
        mock_ts = MagicMock()
        mock_ts.fetch.return_value = mock_hourly_df
        mock_meteostat.hourly.return_value = mock_ts

        with patch.dict("sys.modules", {"meteostat": mock_meteostat}):
            result = weather.fetch_running_mean(
                lat=50.11,
                lon=8.68,
                end_date=date(2025, 6, 30),
                days=7,
            )

        assert isinstance(result, np.ndarray)
        assert result.shape == ()
        assert -20.0 < float(result) < 60.0

    def test_custom_alpha(self, mock_hourly_df: pd.DataFrame, tmp_path, monkeypatch) -> None:
        cache_path = tmp_path / "rm_alpha.parquet"
        monkeypatch.setattr(weather, "_cache_key", lambda *a, **k: cache_path)

        mock_meteostat = MagicMock()
        mock_ts = MagicMock()
        mock_ts.fetch.return_value = mock_hourly_df
        mock_meteostat.hourly.return_value = mock_ts

        with patch.dict("sys.modules", {"meteostat": mock_meteostat}):
            result = weather.fetch_running_mean(
                lat=50.11,
                lon=8.68,
                end_date=date(2025, 6, 30),
                alpha=0.9,
            )
        assert float(result) != np.nan
