"""Tests for local thermal discomfort module (ankle draft, vertical gradient)."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.thermal_local import (
    AnkleDraftResult,
    VerticalGradientResult,
    evaluate_ankle_draft,
    evaluate_vertical_gradient,
    local_discomfort_score,
)


@pytest.fixture
def base_inputs() -> dict:
    return {
        "tdb": np.array([24.0, 25.0, 26.0]),
        "tr": np.array([24.0, 25.0, 26.0]),
        "vr": np.array([0.1, 0.1, 0.1]),
        "rh": np.array([50.0, 50.0, 50.0]),
        "met": 1.2,
        "clo": 0.5,
    }


class TestEvaluateAnkleDraft:
    def test_returns_correct_type(self, base_inputs: dict) -> None:
        result = evaluate_ankle_draft(v_ankle=np.array([0.15, 0.20, 0.25]), **base_inputs)
        assert isinstance(result, AnkleDraftResult)

    def test_ppd_shape_matches_input(self, base_inputs: dict) -> None:
        result = evaluate_ankle_draft(v_ankle=np.array([0.15, 0.20, 0.25]), **base_inputs)
        assert result.ppd_ad.shape == (3,)
        assert result.acceptability.shape == (3,)
        assert result.v_ankle.shape == (3,)

    def test_low_ankle_velocity_acceptable(self, base_inputs: dict) -> None:
        """Low ankle air speed should be acceptable (PPD ≤ 20%)."""
        result = evaluate_ankle_draft(v_ankle=np.array([0.10, 0.10, 0.10]), **base_inputs)
        assert np.all(result.acceptability)
        assert np.all(result.ppd_ad <= 20.0)

    def test_scalar_v_ankle_broadcast(self, base_inputs: dict) -> None:
        result = evaluate_ankle_draft(v_ankle=0.15, **base_inputs)
        assert result.v_ankle.shape == (3,)
        assert np.all(result.v_ankle == 0.15)

    def test_to_dict(self, base_inputs: dict) -> None:
        result = evaluate_ankle_draft(v_ankle=0.15, **base_inputs)
        d = result.to_dict()
        assert "ppd_ad" in d and "acceptability" in d and "v_ankle" in d

    def test_to_json(self, base_inputs: dict) -> None:
        import json

        result = evaluate_ankle_draft(v_ankle=0.15, **base_inputs)
        j = json.loads(result.to_json())
        assert "ppd_ad" in j

    def test_to_dataframe(self, base_inputs: dict) -> None:
        result = evaluate_ankle_draft(v_ankle=0.15, **base_inputs)
        df = result.to_dataframe()
        assert len(df) == 3
        assert "ppd_ad" in df.columns


class TestEvaluateVerticalGradient:
    def test_returns_correct_type(self, base_inputs: dict) -> None:
        result = evaluate_vertical_gradient(
            vertical_tmp_grad=np.array([2.0, 3.0, 4.0]), **base_inputs
        )
        assert isinstance(result, VerticalGradientResult)

    def test_ppd_shape(self, base_inputs: dict) -> None:
        result = evaluate_vertical_gradient(
            vertical_tmp_grad=np.array([2.0, 3.0, 4.0]), **base_inputs
        )
        assert result.ppd_vg.shape == (3,)
        assert result.acceptability.shape == (3,)

    def test_small_gradient_acceptable(self, base_inputs: dict) -> None:
        """Small gradient (≤ 3 °C/m) should be acceptable."""
        result = evaluate_vertical_gradient(
            vertical_tmp_grad=np.array([1.0, 2.0, 3.0]), **base_inputs
        )
        assert np.all(result.acceptability)

    def test_large_gradient_not_acceptable(self, base_inputs: dict) -> None:
        """Large gradient (> 7 °C/m) should not be acceptable."""
        result = evaluate_vertical_gradient(
            vertical_tmp_grad=np.array([8.0, 10.0, 15.0]), **base_inputs
        )
        assert not np.any(result.acceptability)

    def test_scalar_gradient_broadcast(self, base_inputs: dict) -> None:
        result = evaluate_vertical_gradient(vertical_tmp_grad=2.0, **base_inputs)
        assert result.vertical_tmp_grad.shape == (3,)
        assert np.all(result.vertical_tmp_grad == 2.0)


class TestLocalDiscomfortScore:
    def test_both_components(self) -> None:
        score = local_discomfort_score(
            ppd_ad=np.array([15.0, 25.0]),
            ppd_vg=np.array([5.0, 30.0]),
        )
        assert score.shape == (2,)
        # (100-15 + 100-5) / 2 = 90, (100-25 + 100-30) / 2 = 72.5
        np.testing.assert_allclose(score, [90.0, 72.5])

    def test_only_ad(self) -> None:
        score = local_discomfort_score(ppd_ad=np.array([20.0]))
        assert score.shape == (1,)
        np.testing.assert_allclose(score, [80.0])

    def test_only_vg(self) -> None:
        score = local_discomfort_score(ppd_vg=np.array([10.0, 50.0]))
        assert score.shape == (2,)
        np.testing.assert_allclose(score, [90.0, 50.0])

    def test_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            local_discomfort_score()

    def test_clipping(self) -> None:
        """PPD > 100 should clip to 0 score."""
        score = local_discomfort_score(ppd_ad=np.array([150.0]))
        np.testing.assert_allclose(score, [0.0])
