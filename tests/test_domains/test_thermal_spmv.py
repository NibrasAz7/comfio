"""Tests for the simplified PMV (sPMV) thermal comfort domain module."""

from __future__ import annotations

from datetime import date

import numpy as np

from comfio.domains.thermal_spmv import (
    SPMVResult,
    evaluate_spmv,
    spmv_score,
)


class TestEvaluateSPMV:
    def test_returns_result(self, mock_thermal_arrays, n_samples):
        result = evaluate_spmv(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
        )
        assert isinstance(result, SPMVResult)
        assert result.spmv.shape == (n_samples,)
        assert result.score.shape == (n_samples,)

    def test_season_from_date_winter(self, mock_thermal_arrays):
        result = evaluate_spmv(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
            date_ref=date(2025, 1, 15),
        )
        assert result.season == "winter"

    def test_season_from_date_summer(self, mock_thermal_arrays):
        result = evaluate_spmv(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
            date_ref=date(2025, 7, 15),
        )
        assert result.season == "summer"

    def test_season_override(self, mock_thermal_arrays):
        result = evaluate_spmv(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
            season="mid",
        )
        assert result.season == "mid"

    def test_default_season_is_mid(self, mock_thermal_arrays):
        result = evaluate_spmv(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
        )
        assert result.season == "mid"

    def test_score_range_0_100(self, mock_thermal_arrays):
        result = evaluate_spmv(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
        )
        assert np.all(result.score >= 0)
        assert np.all(result.score <= 100)

    def test_neutral_temp_gives_high_score(self):
        # Around 20°C with 50% RH should give sPMV near 0 in mid-season
        temp = np.array([20.0, 20.0, 20.0])
        rh = np.array([50.0, 50.0, 50.0])
        result = evaluate_spmv(temp, rh, season="mid")
        assert np.all(result.score > 50)

    def test_spmv_score_function(self):
        spmv = np.array([0.0, 1.0, -1.0, 3.0, -3.0])
        scores = spmv_score(spmv)
        assert scores[0] == 100.0  # neutral
        assert scores[3] == 0.0  # extreme
        assert scores[4] == 0.0  # extreme

    def test_vapor_pressure_computed(self, mock_thermal_arrays):
        result = evaluate_spmv(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
        )
        assert result.vapor_pressure is not None
        assert np.all(result.vapor_pressure > 0)
