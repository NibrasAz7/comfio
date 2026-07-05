"""Tests for IAQ domain module."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.iaq import (
    CO2_OUTDOOR_BASELINE,
    CO2_THRESHOLDS,
    DEFAULT_CO2_THRESHOLD,
    IAQResult,
    evaluate_iaq,
    iaq_score,
)


class TestEvaluateIAQ:
    def test_default_threshold(self) -> None:
        co2 = np.array([600.0, 1000.0, 1500.0])
        result = evaluate_iaq(co2)
        assert result.threshold_level == DEFAULT_CO2_THRESHOLD
        assert result.threshold_ppm == CO2_THRESHOLDS[DEFAULT_CO2_THRESHOLD]

    def test_compliance(self) -> None:
        co2 = np.array([600.0, 1000.0, 1200.0])
        result = evaluate_iaq(co2, threshold_level="good")
        assert result.compliant[0] == True   # 600 <= 1000
        assert result.compliant[1] == True   # 1000 <= 1000
        assert result.compliant[2] == False  # 1200 > 1000

    def test_result_type(self) -> None:
        result = evaluate_iaq(np.array([800.0]))
        assert isinstance(result, IAQResult)

    def test_all_threshold_levels(self) -> None:
        for level in CO2_THRESHOLDS:
            result = evaluate_iaq(np.array([500.0]), threshold_level=level)  # type: ignore[arg-type]
            assert result.threshold_level == level


class TestIAQScore:
    def test_at_baseline_scores_100(self) -> None:
        score = iaq_score(np.array([CO2_OUTDOOR_BASELINE]), threshold_ppm=1000.0)
        assert score[0] == 100.0

    def test_at_threshold_scores_50(self) -> None:
        score = iaq_score(np.array([1000.0]), threshold_ppm=1000.0)
        assert np.isclose(score[0], 50.0)

    def test_at_double_threshold_scores_0(self) -> None:
        score = iaq_score(np.array([2000.0]), threshold_ppm=1000.0)
        assert score[0] == 0.0

    def test_score_range(self) -> None:
        rng = np.random.default_rng(42)
        co2 = rng.uniform(300, 3000, 100)
        scores = iaq_score(co2, threshold_ppm=1000.0)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 100.0)
