"""Tests for Global IEQ Index integration."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.core.exceptions import DomainNotAvailableError
from comfio.domains.acoustic import evaluate_acoustic
from comfio.domains.iaq import evaluate_iaq
from comfio.domains.thermal import evaluate_thermal
from comfio.domains.visual import evaluate_visual
from comfio.integration.global_ieq import GlobalIEQResult, calculate_global_ieq
from comfio.integration.weights import (
    WeightSchema,
    custom_weights,
    default_weights,
    preset_weights,
)


@pytest.fixture
def all_domain_results(mock_thermal_arrays, mock_visual_array, mock_acoustic_array, mock_iaq_array):
    """All four domain results for integration testing."""
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
    return thermal, visual, acoustic, iaq


class TestCalculateGlobalIEQ:
    def test_all_domains(self, all_domain_results) -> None:
        thermal, visual, acoustic, iaq = all_domain_results
        result = calculate_global_ieq(thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq)
        assert isinstance(result, GlobalIEQResult)
        assert len(result.domains) == 4
        assert result.n_timestamps == 100
        assert np.all(result.index >= 0.0)
        assert np.all(result.index <= 100.0)

    def test_partial_domains(self, all_domain_results) -> None:
        thermal, visual, _, _ = all_domain_results
        result = calculate_global_ieq(thermal=thermal, visual=visual)
        assert len(result.domains) == 2
        assert "thermal" in result.domains
        assert "visual" in result.domains
        # Weights should be renormalized to sum to 1
        total_weight = sum(result.weights_used.values())
        assert np.isclose(total_weight, 1.0)

    def test_single_domain(self, all_domain_results) -> None:
        thermal, _, _, _ = all_domain_results
        result = calculate_global_ieq(thermal=thermal)
        assert len(result.domains) == 1
        # With single domain, IEQ index should equal domain score
        np.testing.assert_allclose(result.index, result.domain_scores["thermal"])

    def test_no_domains_raises(self) -> None:
        with pytest.raises(DomainNotAvailableError):
            calculate_global_ieq()

    def test_custom_weights(self, all_domain_results) -> None:
        thermal, visual, acoustic, iaq = all_domain_results
        weights = custom_weights(0.5, 0.2, 0.1, 0.2)
        result = calculate_global_ieq(
            thermal=thermal,
            visual=visual,
            acoustic=acoustic,
            iaq=iaq,
            weights=weights,
        )
        assert result.weights_used["thermal"] == 0.5

    def test_index_shape_matches_input(self, all_domain_results) -> None:
        thermal, visual, acoustic, iaq = all_domain_results
        result = calculate_global_ieq(thermal=thermal, visual=visual, acoustic=acoustic, iaq=iaq)
        assert result.index.shape == (100,)


class TestWeightSchema:
    def test_default_weights_sum_to_one(self) -> None:
        schema = default_weights()
        assert np.isclose(sum(schema.weights.values()), 1.0)

    def test_all_presets_valid(self) -> None:
        for preset_name in ["default", "equal", "school", "office", "healthcare"]:
            schema = preset_weights(preset_name)  # type: ignore[arg-type]
            assert np.isclose(sum(schema.weights.values()), 1.0)

    def test_custom_weights(self) -> None:
        schema = custom_weights(0.3, 0.3, 0.2, 0.2)
        assert np.isclose(sum(schema.weights.values()), 1.0)
        assert schema.preset_name == "custom"

    def test_invalid_weights_raise(self) -> None:
        from comfio.core.exceptions import WeightConfigurationError

        with pytest.raises(WeightConfigurationError):
            WeightSchema(weights={"thermal": 0.5, "visual": 0.6})

    def test_normalization_with_missing_domain(self) -> None:
        schema = default_weights()
        normed = schema.get_normalized(["thermal", "iaq"])
        assert np.isclose(sum(normed.values()), 1.0)
        # thermal should get more weight than iaq
        assert normed["thermal"] > normed["iaq"]
