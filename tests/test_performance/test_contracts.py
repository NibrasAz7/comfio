"""Tests for performance contracts and compliance reporting."""

from __future__ import annotations

import json

import pytest

from comfio.domains.acoustic import evaluate_acoustic
from comfio.domains.iaq import evaluate_iaq
from comfio.domains.thermal import evaluate_thermal
from comfio.domains.visual import evaluate_visual
from comfio.integration.global_ieq import calculate_global_ieq
from comfio.performance.contract_schema import default_compliance_schema
from comfio.performance.contracts import ComplianceReport, calculate_compliance


@pytest.fixture
def ieq_result_full(mock_thermal_arrays, mock_visual_array, mock_acoustic_array, mock_iaq_array):
    """Full IEQ result for compliance testing."""
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
    return calculate_global_ieq(thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq)


class TestCalculateCompliance:
    def test_report_type(self, ieq_result_full) -> None:
        report = calculate_compliance(ieq_result_full, threshold=80.0)
        assert isinstance(report, ComplianceReport)

    def test_compliance_rate_range(self, ieq_result_full) -> None:
        report = calculate_compliance(ieq_result_full, threshold=80.0)
        assert 0.0 <= report.compliance_rate_pct <= 100.0

    def test_ieq_stats(self, ieq_result_full) -> None:
        report = calculate_compliance(ieq_result_full)
        assert 0.0 <= report.ieq_index_avg <= 100.0
        assert report.ieq_index_min <= report.ieq_index_avg <= report.ieq_index_max
        assert report.ieq_index_std >= 0.0

    def test_domain_compliance(self, ieq_result_full) -> None:
        report = calculate_compliance(ieq_result_full)
        assert "thermal" in report.domain_compliance
        assert "visual" in report.domain_compliance
        assert "acoustic" in report.domain_compliance
        assert "iaq" in report.domain_compliance
        for val in report.domain_compliance.values():
            assert 0.0 <= val <= 100.0

    def test_threshold_effect(self, ieq_result_full) -> None:
        low_threshold = calculate_compliance(ieq_result_full, threshold=50.0)
        high_threshold = calculate_compliance(ieq_result_full, threshold=90.0)
        assert low_threshold.compliance_rate_pct >= high_threshold.compliance_rate_pct

    def test_custom_period(self, ieq_result_full) -> None:
        report = calculate_compliance(
            ieq_result_full,
            period_start=1000000.0,
            period_end=2000000.0,
        )
        assert report.period_start == 1000000.0
        assert report.period_end == 2000000.0


class TestComplianceReportJSON:
    def test_to_json_valid(self, ieq_result_full) -> None:
        report = calculate_compliance(ieq_result_full)
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "ieq_index_avg" in data
        assert "compliance_rate_pct" in data
        assert "domain_compliance" in data

    def test_to_contract_payload(self, ieq_result_full) -> None:
        report = calculate_compliance(ieq_result_full)
        payload = report.to_contract_payload()
        schema = default_compliance_schema()
        # All schema fields should be present in payload
        for field in schema.fields:
            assert field.name in payload

    def test_to_contract_json_valid(self, ieq_result_full) -> None:
        report = calculate_compliance(ieq_result_full)
        json_str = report.to_contract_json()
        data = json.loads(json_str)
        assert "periodStart" in data
        assert "ieqIndexAvg" in data
        assert "complianceRatePct" in data
        assert isinstance(data["thermalCompliant"], bool)
        assert isinstance(data["ieqIndexAvg"], int)


class TestContractSchema:
    def test_default_schema_fields(self) -> None:
        schema = default_compliance_schema()
        assert len(schema.fields) > 0
        assert schema.contract_name == "IEQComplianceOracle"
        assert schema.function_name == "submitCompliance"

    def test_to_abi(self) -> None:
        schema = default_compliance_schema()
        abi = schema.to_abi()
        assert abi["type"] == "function"
        assert abi["name"] == "submitCompliance"
        assert len(abi["inputs"]) == len(schema.fields)

    def test_to_dict(self) -> None:
        schema = default_compliance_schema()
        d = schema.to_dict()
        assert "contract_name" in d
        assert "fields" in d
        assert len(d["fields"]) == len(schema.fields)
