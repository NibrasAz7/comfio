"""Tests for LLM tool schemas and function calling wrappers.

Skipped if pydantic is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from comfio.llm.tools import (
    evaluate_ieq_tool,
    evaluate_thermal_tool,
    get_pydantic_models,
    to_openai_tools,
)


class TestEvaluateThermalTool:
    def test_returns_dict(self) -> None:
        result = evaluate_thermal_tool(
            air_temperature=[23.0, 24.0, 25.0],
            relative_humidity=[50.0, 55.0, 60.0],
        )
        assert isinstance(result, dict)
        assert "mean_pmv" in result
        assert "mean_ppd" in result
        assert "discomfort_risk" in result

    def test_score_range(self) -> None:
        result = evaluate_thermal_tool(
            air_temperature=[23.0] * 10,
            relative_humidity=[50.0] * 10,
        )
        assert -3.0 <= result["mean_pmv"] <= 3.0
        assert 0.0 <= result["mean_ppd"] <= 100.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_thermal_tool(
                air_temperature=[],
                relative_humidity=[50.0],
            )


class TestEvaluateIEQTool:
    def test_returns_dict(self) -> None:
        result = evaluate_ieq_tool(
            air_temperature=[23.0, 24.0, 25.0],
            relative_humidity=[50.0, 55.0, 60.0],
            illuminance=[500.0, 450.0, 480.0],
            noise_laeq=[40.0, 42.0, 38.0],
            co2=[800.0, 700.0, 750.0],
        )
        assert isinstance(result, dict)
        assert "ieq_index_avg" in result
        assert "worst_domain" in result
        assert "markdown" in result
        assert isinstance(result["markdown"], str)

    def test_thermal_only(self) -> None:
        result = evaluate_ieq_tool(
            air_temperature=[23.0, 24.0],
            relative_humidity=[50.0, 55.0],
        )
        assert "ieq_index_avg" in result
        assert "thermal" in result["domains"]

    def test_markdown_contains_domains(self) -> None:
        result = evaluate_ieq_tool(
            air_temperature=[23.0, 24.0],
            relative_humidity=[50.0, 55.0],
            illuminance=[500.0, 450.0],
            noise_laeq=[40.0, 42.0],
            co2=[800.0, 700.0],
        )
        assert "THERMAL" in result["markdown"]
        assert "VISUAL" in result["markdown"]


class TestPydanticModels:
    def test_get_models(self) -> None:
        thermal_model, ieq_model = get_pydantic_models()
        assert thermal_model is not None
        assert ieq_model is not None

    def test_thermal_model_validation(self) -> None:
        thermal_model, _ = get_pydantic_models()
        instance = thermal_model(
            air_temperature=[23.0, 24.0],
            relative_humidity=[50.0, 55.0],
        )
        assert instance.air_temperature == [23.0, 24.0]
        assert instance.air_velocity == 0.1

    def test_ieq_model_defaults(self) -> None:
        _, ieq_model = get_pydantic_models()
        instance = ieq_model(
            air_temperature=[23.0],
            relative_humidity=[50.0],
        )
        assert instance.visual_task_type == "general"
        assert instance.acoustic_nc_level == "NC-35"


class TestOpenAITools:
    def test_returns_list(self) -> None:
        tools = to_openai_tools()
        assert isinstance(tools, list)
        assert len(tools) == 6

    def test_tool_structure(self) -> None:
        tools = to_openai_tools()
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]

    def test_tool_names(self) -> None:
        tools = to_openai_tools()
        names = [t["function"]["name"] for t in tools]
        assert "evaluate_thermal" in names
        assert "evaluate_ieq" in names
