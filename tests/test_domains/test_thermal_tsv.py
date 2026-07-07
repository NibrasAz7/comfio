"""Tests for the TSV processing and CDF augmentation domain module."""

from __future__ import annotations

import numpy as np
import pytest

from comfio.domains.thermal_tsv import (
    TSVResult,
    augment_tsv_cdf,
    evaluate_tsv,
)


class TestAugmentTSVCDF:
    def test_returns_correct_length(
        self, mock_tsv_votes, mock_tsv_timestamps, mock_target_timestamps
    ):
        result = augment_tsv_cdf(
            mock_tsv_votes,
            mock_tsv_timestamps,
            mock_target_timestamps,
        )
        assert len(result) == len(mock_target_timestamps)

    def test_preserves_value_range(
        self, mock_tsv_votes, mock_tsv_timestamps, mock_target_timestamps
    ):
        result = augment_tsv_cdf(
            mock_tsv_votes,
            mock_tsv_timestamps,
            mock_target_timestamps,
        )
        assert result.min() >= mock_tsv_votes.min()
        assert result.max() <= mock_tsv_votes.max()

    def test_preserves_distribution(self, mock_tsv_votes, mock_tsv_timestamps, n_samples):
        # Large target array — distribution should approximate source
        target_ts = np.linspace(0, 100, 1000)
        result = augment_tsv_cdf(mock_tsv_votes, mock_tsv_timestamps, target_ts)
        # Mean of augmented should be close to mean of source
        assert abs(np.mean(result) - np.mean(mock_tsv_votes)) < 0.5

    def test_integer_values(self, mock_tsv_votes, mock_tsv_timestamps, mock_target_timestamps):
        result = augment_tsv_cdf(
            mock_tsv_votes,
            mock_tsv_timestamps,
            mock_target_timestamps,
        )
        # All values should be integers (TSV is ordinal)
        assert np.all(result == np.round(result))

    def test_group_by(self, mock_tsv_votes, mock_tsv_timestamps, mock_target_timestamps):
        groups = np.array(["A"] * 50 + ["B"] * 50)
        result = augment_tsv_cdf(
            mock_tsv_votes,
            mock_tsv_timestamps,
            mock_target_timestamps,
            group_by=groups,
        )
        assert len(result) == len(mock_target_timestamps)

    def test_empty_votes_raises(self, mock_target_timestamps):
        with pytest.raises(ValueError, match="sparse_votes must not be empty"):
            augment_tsv_cdf(np.array([]), np.array([]), mock_target_timestamps)

    def test_mismatched_lengths_raises(self, mock_target_timestamps):
        with pytest.raises(ValueError, match="same length"):
            augment_tsv_cdf(
                np.array([1, 2, 3]),
                np.array([0, 1]),
                mock_target_timestamps,
            )

    def test_single_target(self, mock_tsv_votes, mock_tsv_timestamps):
        result = augment_tsv_cdf(
            mock_tsv_votes,
            mock_tsv_timestamps,
            np.array([5.0]),
        )
        assert len(result) == 1


class TestAugmentTSVCDFTimeAware:
    def test_returns_correct_length(
        self, mock_tsv_votes, mock_tsv_timestamps, mock_target_timestamps
    ):
        result = augment_tsv_cdf(
            mock_tsv_votes,
            mock_tsv_timestamps,
            mock_target_timestamps,
            time_aware=True,
        )
        assert len(result) == len(mock_target_timestamps)

    def test_preserves_value_range(
        self, mock_tsv_votes, mock_tsv_timestamps, mock_target_timestamps
    ):
        result = augment_tsv_cdf(
            mock_tsv_votes,
            mock_tsv_timestamps,
            mock_target_timestamps,
            time_aware=True,
        )
        assert result.min() >= mock_tsv_votes.min()
        assert result.max() <= mock_tsv_votes.max()

    def test_preserves_distribution(self, mock_tsv_votes, mock_tsv_timestamps):
        target_ts = np.linspace(0, 100, 1000)
        result = augment_tsv_cdf(mock_tsv_votes, mock_tsv_timestamps, target_ts, time_aware=True)
        assert abs(np.mean(result) - np.mean(mock_tsv_votes)) < 0.5

    def test_integer_values(self, mock_tsv_votes, mock_tsv_timestamps, mock_target_timestamps):
        result = augment_tsv_cdf(
            mock_tsv_votes,
            mock_tsv_timestamps,
            mock_target_timestamps,
            time_aware=True,
        )
        assert np.all(result == np.round(result))

    def test_temporal_coherence(self):
        # Votes that change over time: cold early, warm late
        votes = np.array([-2, -1, 0, 1, 2], dtype=float)
        vote_ts = np.array([0, 10, 20, 30, 40], dtype=float)
        target_ts = np.arange(0, 41, 1, dtype=float)

        result = augment_tsv_cdf(votes, vote_ts, target_ts, time_aware=True)
        # First 5 targets should be <= last 5 targets (temporal coherence)
        assert np.mean(result[:5]) <= np.mean(result[-5:])

    def test_group_by_time_aware(self, mock_tsv_votes, mock_tsv_timestamps, mock_target_timestamps):
        groups = np.array(["A"] * 50 + ["B"] * 50)
        result = augment_tsv_cdf(
            mock_tsv_votes,
            mock_tsv_timestamps,
            mock_target_timestamps,
            group_by=groups,
            time_aware=True,
        )
        assert len(result) == len(mock_target_timestamps)


class TestEvaluateTSV:
    def test_returns_result(self, mock_tsv_votes):
        result = evaluate_tsv(mock_tsv_votes)
        assert isinstance(result, TSVResult)
        assert result.n_samples == len(mock_tsv_votes)

    def test_compliance(self):
        tsv = np.array([0, 1, -1, 2, -2, 3, -3])
        result = evaluate_tsv(tsv)
        # |TSV| <= 1.5 → compliant
        assert result.compliant[0]  # 0
        assert result.compliant[1]  # 1
        assert not result.compliant[3]  # 2
        assert not result.compliant[5]  # 3

    def test_score_range(self, mock_tsv_votes):
        result = evaluate_tsv(mock_tsv_votes)
        assert np.all(result.score >= 0) and np.all(result.score <= 100)

    def test_neutral_score_100(self):
        result = evaluate_tsv(np.array([0.0, 0.0]))
        assert np.all(result.score == 100.0)

    def test_extreme_score_0(self):
        result = evaluate_tsv(np.array([3.0, -3.0]))
        assert np.all(result.score == 0.0)

    def test_ppd_approx(self):
        result = evaluate_tsv(np.array([0.0, 3.0]))
        assert result.ppd_approx[0] == 5.0  # neutral → 5% PPD
        assert result.ppd_approx[1] == 100.0  # extreme → 100% PPD

    def test_mean_tsv(self):
        tsv = np.array([-1, 0, 1])
        result = evaluate_tsv(tsv)
        assert abs(result.mean_tsv - 0.0) < 0.01

    def test_compliance_rate(self):
        tsv = np.array([0, 1, -1, 2, -2])
        result = evaluate_tsv(tsv)
        # 3 out of 5 compliant
        assert abs(result.compliance_rate - 0.6) < 0.01

    def test_custom_threshold(self):
        tsv = np.array([1, 2, 3])
        result = evaluate_tsv(tsv, compliance_threshold=2.0)
        assert result.compliant[0]  # |1| <= 2
        assert result.compliant[1]  # |2| <= 2
        assert not result.compliant[2]  # |3| > 2
