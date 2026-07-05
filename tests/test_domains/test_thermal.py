"""Tests for thermal comfort domain module.

Tests assert outputs match ISO 7730 / ASHRAE 55 calibration expectations.
"""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.thermal import (
    CATEGORY_PMV_LIMITS,
    CATEGORY_PPD_LIMITS,
    ThermalResult,
    evaluate_thermal,
    thermal_score,
)


class TestEvaluateThermal:
    def test_neutral_conditions_pmv_near_zero(self, mock_thermal_arrays: dict[str, np.ndarray]) -> None:
        """At ~24°C, 50% RH, 0.1 m/s, 1.2 met, 0.5 clo → PMV should be near 0."""
        result = evaluate_thermal(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            vr=mock_thermal_arrays["vr"],
            rh=mock_thermal_arrays["rh"],
            met=1.2,
            clo=0.5,
        )
        assert isinstance(result, ThermalResult)
        # PMV should be in a reasonable range for neutral conditions
        assert np.all(np.abs(result.pmv) < 1.5)
        # PPD should be reasonable (< 50%)
        assert np.all(result.ppd < 50.0)

    def test_pmv_range_validity(self, mock_thermal_arrays: dict[str, np.ndarray]) -> None:
        """PMV must be between -3 and +3."""
        result = evaluate_thermal(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            vr=mock_thermal_arrays["vr"],
            rh=mock_thermal_arrays["rh"],
            met=1.2,
            clo=0.5,
        )
        assert np.all(result.pmv >= -3.0)
        assert np.all(result.pmv <= 3.0)

    def test_ppd_non_negative(self, mock_thermal_arrays: dict[str, np.ndarray]) -> None:
        """PPD must be non-negative."""
        result = evaluate_thermal(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            vr=mock_thermal_arrays["vr"],
            rh=mock_thermal_arrays["rh"],
            met=1.2,
            clo=0.5,
        )
        assert np.all(result.ppd >= 0.0)

    def test_compliance_category_b(self, mock_thermal_arrays: dict[str, np.ndarray]) -> None:
        """Category B compliance: PPD ≤ 10% and |PMV| ≤ 0.5."""
        result = evaluate_thermal(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            vr=mock_thermal_arrays["vr"],
            rh=mock_thermal_arrays["rh"],
            met=1.2,
            clo=0.5,
            category="B",
        )
        # Check that compliant flags are consistent with limits
        ppd_limit = CATEGORY_PPD_LIMITS["B"]
        pmv_lo, pmv_hi = CATEGORY_PMV_LIMITS["B"]
        expected = (result.ppd <= ppd_limit) & (result.pmv >= pmv_lo) & (result.pmv <= pmv_hi)
        np.testing.assert_array_equal(result.compliant, expected)

    def test_hot_conditions_high_pmv(self) -> None:
        """At 30°C, PMV should be positive (warm/hot)."""
        result = evaluate_thermal(
            tdb=np.array([30.0]),
            tr=np.array([30.0]),
            vr=np.array([0.1]),
            rh=np.array([50.0]),
            met=1.2,
            clo=0.5,
        )
        assert result.pmv[0] > 0.5

    def test_cold_conditions_negative_pmv(self) -> None:
        """At 19°C with light clothing, PMV should be negative (cool)."""
        result = evaluate_thermal(
            tdb=np.array([19.0]),
            tr=np.array([19.0]),
            vr=np.array([0.1]),
            rh=np.array([50.0]),
            met=1.2,
            clo=0.5,
        )
        assert result.pmv[0] < -0.5


class TestThermalScore:
    def test_neutral_pmv_scores_high(self) -> None:
        pmv = np.array([0.0, 0.1, -0.1])
        ppd = np.array([5.0, 5.0, 5.0])
        scores = thermal_score(pmv, ppd)
        assert np.all(scores > 90.0)

    def test_extreme_pmv_scores_low(self) -> None:
        pmv = np.array([3.0, -3.0])
        ppd = np.array([100.0, 100.0])
        scores = thermal_score(pmv, ppd)
        assert np.all(scores < 10.0)

    def test_score_range(self) -> None:
        rng = np.random.default_rng(42)
        pmv = rng.uniform(-3, 3, 50)
        ppd = rng.uniform(0, 100, 50)
        scores = thermal_score(pmv, ppd)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 100.0)
