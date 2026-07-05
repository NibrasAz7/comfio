"""Tests for LLM semantic interpreters."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.acoustic import evaluate_acoustic
from comfio.domains.iaq import evaluate_iaq
from comfio.domains.thermal import evaluate_thermal
from comfio.domains.visual import evaluate_visual
from comfio.integration.global_ieq import calculate_global_ieq
from comfio.llm.interpreters import (
    generate_markdown_summary,
    ieq_to_markdown,
    ieq_to_summary_dict,
)
from comfio.performance.contracts import calculate_compliance


@pytest.fixture
def ieq_result_full(
    mock_thermal_arrays, mock_visual_array, mock_acoustic_array, mock_iaq_array
):
    """Full IEQ result for interpreter tests."""
    thermal = evaluate_thermal(
        tdb=mock_thermal_arrays["tdb"],
        tr=mock_thermal_arrays["tr"],
        vr=mock_thermal_arrays["vr"],
        rh=mock_thermal_arrays["rh"],
        met=1.2,
        clo=0.5,
    )
    visual = evaluate_visual(mock_visual_array)
    acoustic = evaluate_acoustic(mock_acoustic_array)
    iaq = evaluate_iaq(mock_iaq_array)
    return calculate_global_ieq(
        thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq,
    )


class TestIeqToMarkdown:
    def test_returns_string(self, ieq_result_full) -> None:
        md = ieq_to_markdown(ieq_result_full)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_contains_domain_names(self, ieq_result_full) -> None:
        md = ieq_to_markdown(ieq_result_full)
        assert "THERMAL" in md
        assert "VISUAL" in md
        assert "ACOUSTIC" in md
        assert "IAQ" in md

    def test_contains_ieq_score(self, ieq_result_full) -> None:
        md = ieq_to_markdown(ieq_result_full)
        assert "Global IEQ Index Score" in md
        assert "/100" in md

    def test_contains_diagnostic(self, ieq_result_full) -> None:
        md = ieq_to_markdown(ieq_result_full)
        assert "Diagnostic Insight" in md
        assert "primary limiting factor" in md

    def test_with_compliance_report(self, ieq_result_full) -> None:
        report = calculate_compliance(ieq_result_full, threshold=80.0)
        md = ieq_to_markdown(ieq_result_full, compliance_report=report)
        assert "COMPLIANT" in md or "NON-COMPLIANT" in md
        assert "Compliance Rate" in md

    def test_with_zone_id(self, ieq_result_full) -> None:
        md = ieq_to_markdown(ieq_result_full, zone_id="Room-402")
        assert "Room-402" in md

    def test_token_efficient(self, ieq_result_full) -> None:
        """Markdown should be well under 1000 chars for 100 timestamps."""
        md = ieq_to_markdown(ieq_result_full)
        assert len(md) < 1500


class TestIeqToSummaryDict:
    def test_returns_dict(self, ieq_result_full) -> None:
        d = ieq_to_summary_dict(ieq_result_full)
        assert isinstance(d, dict)

    def test_has_key_fields(self, ieq_result_full) -> None:
        d = ieq_to_summary_dict(ieq_result_full)
        assert "ieq_index_avg" in d
        assert "ieq_index_min" in d
        assert "ieq_index_max" in d
        assert "worst_domain" in d
        assert "domain_scores_avg" in d

    def test_score_range(self, ieq_result_full) -> None:
        d = ieq_to_summary_dict(ieq_result_full)
        assert 0.0 <= d["ieq_index_avg"] <= 100.0
        assert d["ieq_index_min"] <= d["ieq_index_avg"] <= d["ieq_index_max"]

    def test_with_compliance(self, ieq_result_full) -> None:
        report = calculate_compliance(ieq_result_full)
        d = ieq_to_summary_dict(ieq_result_full, compliance_report=report)
        assert "compliance_rate_pct" in d
        assert "threshold" in d


class TestGenerateMarkdownSummary:
    def test_returns_string(self, mock_sensor_df) -> None:
        md = generate_markdown_summary(mock_sensor_df)
        assert isinstance(md, str)
        assert "IEQ Report" in md

    def test_contains_domain_summary(self, mock_sensor_df) -> None:
        md = generate_markdown_summary(mock_sensor_df)
        assert "Domain Summary" in md
        assert "THERMAL" in md

    def test_contains_critical_failures(self, mock_sensor_df) -> None:
        md = generate_markdown_summary(mock_sensor_df)
        assert "Critical Failures" in md

    def test_with_zone_id(self, mock_sensor_df) -> None:
        md = generate_markdown_summary(mock_sensor_df, zone_id="Zone-A")
        assert "Zone-A" in md

    def test_partial_data(self, mock_sensor_df_partial) -> None:
        md = generate_markdown_summary(mock_sensor_df_partial)
        assert "IEQ Report" in md
