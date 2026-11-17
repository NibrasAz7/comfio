"""Unit tests for the certification-preset checks (pure logic)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from comfio.gui._certify import reset_air, well_t06


class TestResetAir:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=100, freq="h"),
                "pm25": [10.0] * 90 + [np.nan] * 10,
                "tvoc": [400.0] * 95 + [np.nan] * 5,
                "co2": [550.0] * 100,
                "rh": [50.0] * 100,
            }
        )

    def _map(self) -> dict[str, str]:
        return {
            "pm25_ugm3": "pm25",
            "tvoc_ugm3": "tvoc",
            "co2_ppm": "co2",
            "relative_humidity_pct": "rh",
        }

    def test_all_pass(self) -> None:
        out = reset_air(self._df(), self._map(), "timestamp")
        assert out["criteria_passed"] == 4
        assert out["missing"] == []
        assert out["duration_days"] == pytest.approx(4.125)

    def test_missing_input_reported(self) -> None:
        m = self._map()
        del m["tvoc_ugm3"]
        out = reset_air(self._df(), m, "timestamp")
        assert "TVOC" in out["missing"]
        assert out["criteria_passed"] == 3

    def test_low_coverage_fails(self) -> None:
        df = self._df()
        df["pm25"] = np.nan  # 0% coverage
        out = reset_air(df, self._map(), "timestamp")
        pm25_row = next(c for c in out["criteria"] if c["criterion"] == "PM2.5")
        assert pm25_row["status"] == "fail"

    def test_out_of_range_fails(self) -> None:
        df = self._df()
        df["rh"] = 80.0
        out = reset_air(df, self._map(), "timestamp")
        rh_row = next(c for c in out["criteria"] if c["criterion"] == "Relative humidity")
        assert rh_row["status"] == "fail"
        assert rh_row["within target %"] == 0.0

    def test_no_timestamp_caveat(self) -> None:
        out = reset_air(self._df(), self._map(), None)
        assert out["duration_days"] is None
        assert any("timestamp" in c.lower() for c in out["caveats"])


class TestWellT06:
    def test_pass_and_fail(self) -> None:
        ok = SimpleNamespace(pmv=np.array([0.1] * 95 + [0.8] * 5))
        assert well_t06(ok)["criteria"][0]["status"] == "pass"
        bad = SimpleNamespace(pmv=np.array([0.1] * 50 + [1.2] * 50))
        assert well_t06(bad)["criteria"][0]["status"] == "fail"

    def test_nan_excluded_from_fraction(self) -> None:
        res = SimpleNamespace(pmv=np.array([0.0] * 9 + [np.nan]))
        out = well_t06(res)["criteria"][0]
        assert out["within target %"] == 100.0
        assert out["coverage %"] == 90.0

    def test_unavailable_without_thermal(self) -> None:
        out = well_t06(None)
        assert out["available"] is False
