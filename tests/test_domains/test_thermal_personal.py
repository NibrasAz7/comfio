"""Tests for the personalised thermal comfort domain module."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.thermal_personal import (
    PersonalisationIndex,
    PersonalisedAdaptiveResult,
    PersonalisedPMVResult,
    SeasonalPersonalisationIndex,
    evaluate_personalised_adaptive,
    evaluate_personalised_pmv,
    evaluate_personalised_spmv,
    train_personalisation,
    train_seasonal_personalisation,
)


class TestTrainPersonalisation:
    def test_returns_index(self, mock_pmv_tsv_pairs):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_personalisation(pmv, tsv)
        assert isinstance(idx, PersonalisationIndex)
        assert idx.n_samples == len(pmv)

    def test_alpha_close_to_true(self, mock_pmv_tsv_pairs):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_personalisation(pmv, tsv)
        # True alpha was 1.1 — should be recovered approximately
        assert abs(idx.alpha - 1.1) < 0.3

    def test_beta_close_to_true(self, mock_pmv_tsv_pairs):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_personalisation(pmv, tsv)
        # True beta was 0.2
        assert abs(idx.beta - 0.2) < 0.3

    def test_r_squared_positive(self, mock_pmv_tsv_pairs):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_personalisation(pmv, tsv)
        assert idx.r_squared > 0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            train_personalisation(np.array([1.0]), np.array([1.0]))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            train_personalisation(np.array([1, 2, 3]), np.array([1, 2]))

    def test_apply(self, mock_pmv_tsv_pairs):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_personalisation(pmv, tsv)
        personalised = idx.apply(pmv)
        assert personalised.shape == pmv.shape

    def test_few_samples_warns(self):
        pmv = np.array([0.5, 0.6, 0.7])
        tsv = np.array([1, 1, 1])
        with pytest.warns(UserWarning, match="Only 3 samples"):
            train_personalisation(pmv, tsv)


class TestTrainSeasonalPersonalisation:
    def test_returns_seasonal_index(self, mock_pmv_tsv_pairs, mock_seasonal_dates):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_seasonal_personalisation(pmv, tsv, mock_seasonal_dates)
        assert isinstance(idx, SeasonalPersonalisationIndex)
        assert len(idx.indices) > 0

    def test_get_index_for_season(self, mock_pmv_tsv_pairs, mock_seasonal_dates):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_seasonal_personalisation(pmv, tsv, mock_seasonal_dates)
        # Should have at least one season
        season = list(idx.indices.keys())[0]
        assert isinstance(idx.get_index(season), PersonalisationIndex)

    def test_missing_season_raises(self, mock_pmv_tsv_pairs, mock_seasonal_dates):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_seasonal_personalisation(pmv, tsv, mock_seasonal_dates)
        with pytest.raises(KeyError, match="No personalisation index"):
            idx.get_index("nonexistent")


class TestEvaluatePersonalisedPMV:
    def test_returns_result(self, mock_thermal_arrays, mock_pmv_tsv_pairs):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_personalisation(pmv, tsv)
        result = evaluate_personalised_pmv(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            vr=mock_thermal_arrays["vr"],
            rh=mock_thermal_arrays["rh"],
            met=1.2,
            clo=0.5,
            personalisation_index=idx,
        )
        assert isinstance(result, PersonalisedPMVResult)
        assert result.personalised_pmv.shape == mock_thermal_arrays["tdb"].shape
        assert result.alpha == idx.alpha
        assert result.beta == idx.beta


class TestEvaluatePersonalisedSPMV:
    def test_returns_result(self, mock_thermal_arrays, mock_pmv_tsv_pairs):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_personalisation(pmv, tsv)
        result = evaluate_personalised_spmv(
            indoor_temp=mock_thermal_arrays["tdb"],
            indoor_rh=mock_thermal_arrays["rh"],
            personalisation_index=idx,
        )
        assert isinstance(result, PersonalisedPMVResult)
        assert result.personalised_pmv.shape == mock_thermal_arrays["tdb"].shape


class TestEvaluatePersonalisedAdaptive:
    def test_returns_result_ashrae(self, mock_thermal_arrays, mock_pmv_tsv_pairs):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_personalisation(pmv, tsv)
        result = evaluate_personalised_adaptive(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_outdoor=20.0,
            personalisation_index=idx,
            standard="ashrae",
        )
        assert isinstance(result, PersonalisedAdaptiveResult)
        assert result.adaptive_result.standard == "ashrae"

    def test_returns_result_en(self, mock_thermal_arrays, mock_pmv_tsv_pairs):
        pmv, tsv = mock_pmv_tsv_pairs
        idx = train_personalisation(pmv, tsv)
        result = evaluate_personalised_adaptive(
            tdb=mock_thermal_arrays["tdb"],
            tr=mock_thermal_arrays["tr"],
            t_outdoor=20.0,
            personalisation_index=idx,
            standard="en",
        )
        assert isinstance(result, PersonalisedAdaptiveResult)
        assert result.adaptive_result.standard == "en"
