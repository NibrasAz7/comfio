"""Tests for ResultBase mixin (to_dict, to_json, to_dataframe)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from comfio import (
    evaluate_thermal,
    evaluate_iaq,
    evaluate_visual,
    evaluate_acoustic,
)
from comfio.core.result_base import ResultBase, is_result_instance


@pytest.fixture
def thermal_result():
    return evaluate_thermal(
        tdb=np.array([24.0, 25.0, 26.0]),
        tr=np.array([24.0, 25.0, 26.0]),
        vr=np.array([0.1, 0.1, 0.1]),
        rh=np.array([50.0, 50.0, 50.0]),
        met=1.2,
        clo=0.5,
    )


@pytest.fixture
def iaq_result():
    return evaluate_iaq(co2=np.array([400.0, 800.0, 1200.0]), threshold_level="good")


class TestToDict:
    def test_thermal_to_dict(self, thermal_result) -> None:
        d = thermal_result.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {"pmv", "ppd", "compliant", "category"}
        assert isinstance(d["pmv"], np.ndarray)

    def test_iaq_to_dict(self, iaq_result) -> None:
        d = iaq_result.to_dict()
        assert set(d.keys()) == {"co2", "threshold_ppm", "compliant", "score", "threshold_level"}


class TestToJson:
    def test_valid_json(self, thermal_result) -> None:
        j = thermal_result.to_json()
        parsed = json.loads(j)
        assert "pmv" in parsed
        assert isinstance(parsed["pmv"], list)
        assert len(parsed["pmv"]) == 3

    def test_custom_indent(self, thermal_result) -> None:
        j = thermal_result.to_json(indent=None)
        assert "\n" not in j

    def test_numpy_scalars_serialized(self, iaq_result) -> None:
        """Numpy float scalars should serialize to Python floats."""
        j = json.loads(iaq_result.to_json())
        assert isinstance(j["threshold_ppm"], (int, float))


class TestToDataFrame:
    def test_thermal_to_dataframe(self, thermal_result) -> None:
        df = thermal_result.to_dataframe()
        assert len(df) == 3
        assert "pmv" in df.columns
        assert "ppd" in df.columns
        assert "compliant" in df.columns
        # Scalar field broadcast
        assert "category" in df.columns
        assert (df["category"] == "B").all()

    def test_iaq_to_dataframe(self, iaq_result) -> None:
        df = iaq_result.to_dataframe()
        assert len(df) == 3
        assert "co2" in df.columns
        assert "score" in df.columns


class TestIsResultInstance:
    def test_true_for_result(self, thermal_result) -> None:
        assert is_result_instance(thermal_result)

    def test_false_for_plain_dict(self) -> None:
        assert not is_result_instance({"pmv": 0})

    def test_false_for_class(self) -> None:
        from comfio.domains.thermal import ThermalResult

        assert not is_result_instance(ThermalResult)


class TestResultBaseAllResults:
    """Verify that all major Result types inherit ResultBase."""

    def test_thermal_inherits(self, thermal_result) -> None:
        assert isinstance(thermal_result, ResultBase)

    def test_iaq_inherits(self, iaq_result) -> None:
        assert isinstance(iaq_result, ResultBase)

    def test_visual_inherits(self) -> None:
        result = evaluate_visual(illuminance=np.array([300.0, 500.0]), task_type="office")
        assert isinstance(result, ResultBase)

    def test_acoustic_inherits(self) -> None:
        result = evaluate_acoustic(laeq=np.array([35.0, 45.0]))
        assert isinstance(result, ResultBase)
