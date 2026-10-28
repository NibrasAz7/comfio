"""Tests for the simplified PMV (sPMV) thermal comfort domain module."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from comfio.domains.thermal_spmv import (
    ICL_GAM_COEFFS,
    ICL_GAM_CONST,
    ICL_RANGES,
    SPMVOPT_COEFFS,
    SPMVOptResult,
    SPMVResult,
    _magnus_vapor_pressure,
    evaluate_spmv,
    evaluate_spmvopt,
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


class TestEvaluateSPMVOpt:
    """sPMVopt — Morresi et al. (2026) optimized simplified PMV."""

    def test_returns_result(self, mock_thermal_arrays, mock_outdoor_temp):
        result = evaluate_spmvopt(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
            outdoor_temp=mock_outdoor_temp,
            climate="portugal",
        )
        assert isinstance(result, SPMVOptResult)
        assert result.spmvopt.shape == result.indoor_temp.shape
        assert result.icl_estimated.shape == result.indoor_temp.shape
        assert result.icl_band.shape == result.indoor_temp.shape
        assert result.score.shape == result.indoor_temp.shape
        assert result.climate == "portugal"

    @pytest.mark.parametrize("climate", ["france", "portugal", "sweden"])
    def test_all_climates(self, climate, mock_thermal_arrays, mock_outdoor_temp):
        result = evaluate_spmvopt(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
            outdoor_temp=mock_outdoor_temp,
            climate=climate,
        )
        assert result.climate == climate
        assert np.all(np.isfinite(result.spmvopt))
        assert np.all(result.score >= 0) and np.all(result.score <= 100)

    def test_bad_climate_raises(self, mock_thermal_arrays, mock_outdoor_temp):
        with pytest.raises(KeyError):
            evaluate_spmvopt(
                indoor_temp=mock_thermal_arrays["tdb"],
                indoor_rh=mock_thermal_arrays["rh"],
                outdoor_temp=mock_outdoor_temp,
                climate="italy",
            )

    def test_icl_estimation_within_paper_range(self, mock_thermal_arrays, mock_outdoor_temp):
        result = evaluate_spmvopt(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
            outdoor_temp=mock_outdoor_temp,
        )
        # Icl is clipped to the paper's band table support [0.25, 1.65] clo
        assert np.all(result.icl_estimated >= 0.25)
        assert np.all(result.icl_estimated <= 1.65)
        assert np.all(result.icl_band >= 0) and np.all(result.icl_band <= 3)

    def test_hand_computed_single_sample(self):
        """Verify the full pipeline on one hand-computable sample."""
        indoor_temp = np.array([22.0])
        indoor_rh = np.array([50.0])
        outdoor_temp = np.array([15.0])

        # Step 1 (paper Eq. 11): Icl = f(Tout)+f(Ta)+f(RH)+const
        c = ICL_GAM_COEFFS
        icl = (
            c["tout"]["c1"] * 15.0 + c["tout"]["c2"] * 15.0**2 + c["tout"]["c3"] * 15.0**3
            + c["ta"]["c1"] * 22.0 + c["ta"]["c2"] * 22.0**2 + c["ta"]["c3"] * 22.0**3
            + c["rh"]["c1"] * 50.0
            + ICL_GAM_CONST
        )
        icl = np.clip(icl, 0.25, 1.65)
        assert 0.25 <= icl <= 1.65
        band = next(i for i, (lo, hi) in enumerate(ICL_RANGES) if lo <= icl <= hi)

        # Step 2: sPMVopt = a*Ta + b*pv - c (Portugal coefficients)
        pv = _magnus_vapor_pressure(indoor_temp, indoor_rh)[0]
        co = SPMVOPT_COEFFS["portugal"][band]
        expected = co["a"] * 22.0 + co["b"] * pv - co["c"]

        result = evaluate_spmvopt(
            indoor_temp=indoor_temp,
            indoor_rh=indoor_rh,
            outdoor_temp=outdoor_temp,
            climate="portugal",
        )
        assert result.icl_estimated[0] == pytest.approx(icl, abs=1e-6)
        assert result.icl_band[0] == band
        assert result.spmvopt[0] == pytest.approx(expected, abs=1e-6)
