"""Tests for acoustic comfort domain module."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.acoustic import (
    DEFAULT_NC_LEVEL,
    NC_THRESHOLDS,
    AcousticResult,
    acoustic_score,
    evaluate_acoustic,
)


class TestEvaluateAcoustic:
    def test_default_nc_level(self) -> None:
        laeq = np.array([35.0, 41.0, 50.0])
        result = evaluate_acoustic(laeq)
        assert result.nc_level == DEFAULT_NC_LEVEL
        assert result.threshold_db == NC_THRESHOLDS[DEFAULT_NC_LEVEL]

    def test_compliance(self) -> None:
        laeq = np.array([30.0, 41.0, 50.0])
        result = evaluate_acoustic(laeq, nc_level="NC-35")
        assert result.compliant[0] == True   # 30 <= 41
        assert result.compliant[1] == True   # 41 <= 41
        assert result.compliant[2] == False  # 50 > 41

    def test_result_type(self) -> None:
        result = evaluate_acoustic(np.array([40.0]))
        assert isinstance(result, AcousticResult)

    def test_all_nc_levels(self) -> None:
        for nc_level in NC_THRESHOLDS:
            result = evaluate_acoustic(np.array([30.0]), nc_level=nc_level)  # type: ignore[arg-type]
            assert result.nc_level == nc_level


class TestAcousticScore:
    def test_well_below_threshold(self) -> None:
        # threshold=41, laeq=31 → (41+10-31)/20 = 1.0 → 100
        score = acoustic_score(np.array([31.0]), threshold_db=41.0)
        assert score[0] == 100.0

    def test_at_threshold(self) -> None:
        # threshold=41, laeq=41 → (41+10-41)/20 = 0.5 → 50
        score = acoustic_score(np.array([41.0]), threshold_db=41.0)
        assert score[0] == 50.0

    def test_well_above_threshold(self) -> None:
        # threshold=41, laeq=51 → (41+10-51)/20 = 0 → 0
        score = acoustic_score(np.array([51.0]), threshold_db=41.0)
        assert score[0] == 0.0

    def test_score_range(self) -> None:
        rng = np.random.default_rng(42)
        laeq = rng.uniform(20, 70, 100)
        scores = acoustic_score(laeq, threshold_db=41.0)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 100.0)
