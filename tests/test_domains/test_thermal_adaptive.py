"""Tests for the adaptive thermal comfort domain module."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.thermal_adaptive import (
    AdaptiveThermalResult,
    adaptive_thermal_score,
    evaluate_adaptive_ashrae,
    evaluate_adaptive_en,
)


class TestEvaluateAdaptiveASHRAE:
    def test_returns_result(self, mock_thermal_arrays, n_samples):
        result = evaluate_adaptive_ashrae(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_prevail=20.0,
        )
        assert isinstance(result, AdaptiveThermalResult)
        assert result.standard == "ashrae"
        assert result.t_op.shape == (n_samples,)
        assert result.compliant.shape == (n_samples,)
        assert result.score.shape == (n_samples,)

    def test_comfort_temperature(self, mock_thermal_arrays):
        result = evaluate_adaptive_ashrae(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_prevail=20.0,
        )
        # t_comf = 0.31 * 20 + 17.8 = 24.0
        assert abs(result.t_comf - 24.0) < 0.01

    def test_80_pct_band(self, mock_thermal_arrays):
        result = evaluate_adaptive_ashrae(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_prevail=20.0,
            acceptability=80,
        )
        assert abs((result.t_comf_upper - result.t_comf_lower) - 7.0) < 0.01  # ±3.5

    def test_90_pct_band(self, mock_thermal_arrays):
        result = evaluate_adaptive_ashrae(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_prevail=20.0,
            acceptability=90,
        )
        assert abs((result.t_comf_upper - result.t_comf_lower) - 5.0) < 0.01  # ±2.5

    def test_out_of_range_raises(self, mock_thermal_arrays):
        with pytest.raises(ValueError, match="outside.*ASHRAE"):
            evaluate_adaptive_ashrae(
                tdb=mock_thermal_arrays["tdb"],
                tr=mock_thermal_arrays["tr"],
                t_prevail=5.0,
            )

    def test_score_range(self, mock_thermal_arrays):
        result = evaluate_adaptive_ashrae(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_prevail=20.0,
        )
        assert np.all(result.score >= 0) and np.all(result.score <= 100)


class TestEvaluateAdaptiveEN:
    def test_returns_result(self, mock_thermal_arrays, n_samples):
        result = evaluate_adaptive_en(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_running_mean=20.0,
        )
        assert isinstance(result, AdaptiveThermalResult)
        assert result.standard == "en"
        assert result.t_op.shape == (n_samples,)

    def test_comfort_temperature(self, mock_thermal_arrays):
        result = evaluate_adaptive_en(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_running_mean=20.0,
        )
        # t_comf = 0.33 * 20 + 18.8 = 25.4
        assert abs(result.t_comf - 25.4) < 0.01

    def test_category_ii_band(self, mock_thermal_arrays):
        result = evaluate_adaptive_en(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_running_mean=20.0,
            category="ii",
        )
        assert abs((result.t_comf_upper - result.t_comf_lower) - 6.0) < 0.01  # ±3

    def test_out_of_range_raises(self, mock_thermal_arrays):
        with pytest.raises(ValueError, match="outside.*EN"):
            evaluate_adaptive_en(
                tdb=mock_thermal_arrays["tdb"],
                tr=mock_thermal_arrays["tr"],
                t_running_mean=35.0,
            )


class TestAdaptiveThermalScore:
    def test_score_at_comfort_temp(self):
        score = adaptive_thermal_score(
            np.array([24.0]), 24.0, 20.5, 27.5,
        )
        assert score[0] == 100.0

    def test_score_at_boundary(self):
        score = adaptive_thermal_score(
            np.array([27.5]), 24.0, 20.5, 27.5,
        )
        assert abs(score[0] - 50.0) < 0.01

    def test_score_far_outside(self):
        score = adaptive_thermal_score(
            np.array([35.0]), 24.0, 20.5, 27.5,
        )
        assert score[0] == 0.0
