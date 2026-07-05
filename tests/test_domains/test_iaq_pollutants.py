"""Tests for the IAQ pollutant evaluation domain module."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.iaq_pollutants import (
    PollutantIAQResult,
    evaluate_iaq_pollutants,
    pollutant_iaq_score,
)


class TestEvaluateIAQPollutants:
    def test_returns_result_with_pm25(self, mock_pollutant_arrays, n_samples):
        result = evaluate_iaq_pollutants(pm25=mock_pollutant_arrays["pm25"])
        assert isinstance(result, PollutantIAQResult)
        assert result.pm25 is not None
        assert len(result.pm25) == n_samples
        assert result.score.shape == (n_samples,)

    def test_all_pollutants(self, mock_pollutant_arrays, n_samples):
        result = evaluate_iaq_pollutants(**mock_pollutant_arrays)
        assert result.pm25 is not None
        assert result.pm10 is not None
        assert result.tvoc is not None
        assert result.formaldehyde is not None
        assert result.co is not None
        assert result.score.shape == (n_samples,)

    def test_score_range_0_100(self, mock_pollutant_arrays):
        result = evaluate_iaq_pollutants(**mock_pollutant_arrays)
        assert np.all(result.score >= 0)
        assert np.all(result.score <= 100)

    def test_excellent_conditions_score_high(self):
        pm25 = np.array([3.0, 4.0, 5.0])
        result = evaluate_iaq_pollutants(pm25=pm25, threshold_level="excellent")
        assert np.all(result.score >= 90)

    def test_poor_conditions_score_low(self):
        pm25 = np.array([60.0, 80.0, 100.0])
        result = evaluate_iaq_pollutants(pm25=pm25)
        assert np.all(result.score < 10)

    def test_compliance_flags(self, mock_pollutant_arrays):
        result = evaluate_iaq_pollutants(**mock_pollutant_arrays)
        assert result.compliant_pm25 is not None
        assert result.compliant_pm25.dtype == bool

    def test_no_pollutants_raises(self):
        with pytest.raises(ValueError, match="At least one pollutant"):
            evaluate_iaq_pollutants()

    def test_pollutant_iaq_score_convenience(self, mock_pollutant_arrays, n_samples):
        scores = pollutant_iaq_score(pm25=mock_pollutant_arrays["pm25"])
        assert scores.shape == (n_samples,)
        assert np.all(scores >= 0) and np.all(scores <= 100)

    def test_threshold_level_affects_compliance(self):
        pm25 = np.array([40.0, 40.0, 40.0])
        result_good = evaluate_iaq_pollutants(pm25=pm25, threshold_level="good")
        result_poor = evaluate_iaq_pollutants(pm25=pm25, threshold_level="poor")
        # At 40 µg/m³, non-compliant for "good" (15) but compliant for "poor" (55)
        assert not np.all(result_good.compliant_pm25)
        assert np.all(result_poor.compliant_pm25)
