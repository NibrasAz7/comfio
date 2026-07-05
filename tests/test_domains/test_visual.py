"""Tests for visual comfort domain module.

Tests assert outputs match EN 12464-1 illuminance targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.visual import (
    ILLUMINANCE_TARGETS,
    UGR_LIMITS,
    VisualResult,
    evaluate_visual,
    visual_score,
)


class TestEvaluateVisual:
    def test_office_writing_target_500(self) -> None:
        lux = np.array([500.0, 600.0, 400.0])
        result = evaluate_visual(lux, task_type="office_writing")
        assert result.target_lux == 500.0
        assert result.compliant[0] == True   # 500 >= 500
        assert result.compliant[1] == True   # 600 >= 500
        assert result.compliant[2] == False  # 400 < 500

    def test_general_target(self) -> None:
        lux = np.array([500.0])
        result = evaluate_visual(lux, task_type="general")
        assert result.target_lux == ILLUMINANCE_TARGETS["general"]

    def test_with_ugr(self) -> None:
        lux = np.array([500.0, 500.0])
        ugr = np.array([15.0, 25.0])
        result = evaluate_visual(lux, task_type="office_writing", ugr=ugr)
        assert result.ugr_compliant is not None
        assert result.ugr_compliant[0] == True   # 15 <= 19
        assert result.ugr_compliant[1] == False  # 25 > 19

    def test_without_ugr(self) -> None:
        lux = np.array([500.0])
        result = evaluate_visual(lux)
        assert result.ugr_compliant is None

    def test_result_type(self) -> None:
        result = evaluate_visual(np.array([500.0]))
        assert isinstance(result, VisualResult)


class TestVisualScore:
    def test_at_target_scores_high(self) -> None:
        score = visual_score(np.array([500.0]), target_lux=500.0)
        assert score[0] == 100.0

    def test_under_lit_scores_low(self) -> None:
        score = visual_score(np.array([100.0]), target_lux=500.0)
        assert score[0] < 30.0

    def test_over_lit_penalized(self) -> None:
        score = visual_score(np.array([1000.0]), target_lux=500.0)
        # Over-lit: 100 - 20*(2-1) = 80
        assert score[0] == 80.0

    def test_score_range(self) -> None:
        rng = np.random.default_rng(42)
        lux = rng.uniform(0, 2000, 100)
        scores = visual_score(lux, target_lux=500.0)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 100.0)

    def test_ugr_component(self) -> None:
        # With UGR: 0.7 * illuminance_score + 0.3 * ugr_score
        lux = np.array([500.0])
        ugr = np.array([19.0])  # at limit
        score = visual_score(lux, target_lux=500.0, ugr=ugr, ugr_limit=19.0)
        # illuminance_score = 100 (at target)
        # ugr_score = 100 * (30 - 19) / 20 = 55
        expected = 0.7 * 100.0 + 0.3 * 55.0
        assert np.isclose(score[0], expected)
