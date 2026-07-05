"""Tests for Global IEQ Index integration with pollutant_iaq and tsv parameters."""

from __future__ import annotations

import numpy as np

from comfio.domains.iaq_pollutants import evaluate_iaq_pollutants
from comfio.domains.thermal_tsv import evaluate_tsv
from comfio.integration.global_ieq import calculate_global_ieq


class TestGlobalIEQWithPollutants:
    def test_pollutant_only_maps_to_iaq(self, mock_pollutant_arrays, n_samples):
        pollutant_res = evaluate_iaq_pollutants(**mock_pollutant_arrays)
        result = calculate_global_ieq(pollutant_iaq=pollutant_res)
        assert "iaq" in result.domain_scores
        assert result.n_timestamps == n_samples

    def test_pollutant_and_iaq_blend(self, mock_pollutant_arrays, mock_iaq_array, n_samples):
        from comfio.domains.iaq import evaluate_iaq

        iaq_res = evaluate_iaq(co2=mock_iaq_array)
        pollutant_res = evaluate_iaq_pollutants(**mock_pollutant_arrays)
        result = calculate_global_ieq(iaq=iaq_res, pollutant_iaq=pollutant_res)
        assert "iaq" in result.domain_scores
        # Score should be a blend of both
        blended = 0.5 * iaq_res.score + 0.5 * pollutant_res.score
        np.testing.assert_allclose(result.domain_scores["iaq"], blended)


class TestGlobalIEQWithTSV:
    def test_tsv_only_maps_to_thermal(self, mock_tsv_votes):
        tsv_res = evaluate_tsv(mock_tsv_votes)
        result = calculate_global_ieq(tsv=tsv_res)
        assert "thermal" in result.domain_scores
        assert result.n_timestamps == len(mock_tsv_votes)

    def test_tsv_overrides_thermal(self, mock_thermal_arrays, mock_tsv_votes):
        from comfio.domains.thermal import evaluate_thermal

        # Need same length arrays
        n = 10
        thermal_res = evaluate_thermal(
            tdb=mock_thermal_arrays["tdb"][:n],
            tr=mock_thermal_arrays["tr"][:n],
            vr=mock_thermal_arrays["vr"][:n],
            rh=mock_thermal_arrays["rh"][:n],
            met=1.2,
            clo=0.5,
        )
        tsv_res = evaluate_tsv(mock_tsv_votes)
        result = calculate_global_ieq(thermal=thermal_res, tsv=tsv_res)
        # TSV score should override thermal score
        np.testing.assert_allclose(result.domain_scores["thermal"], tsv_res.score)
