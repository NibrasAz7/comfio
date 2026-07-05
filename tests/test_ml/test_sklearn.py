"""Tests for scikit-learn IEQ transformers.

Skipped if scikit-learn is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from comfio.ml.sklearn_transformers import IEQFeatureExtractor


class TestIEQFeatureExtractor:
    def test_fit_transform(self, mock_sensor_df: pd.DataFrame) -> None:
        extractor = IEQFeatureExtractor()
        extractor.fit(mock_sensor_df)
        features = extractor.transform(mock_sensor_df)
        assert features.shape[0] == len(mock_sensor_df)
        # Should have ieq_index + 4 domain scores = 5 features
        assert features.shape[1] == 5

    def test_feature_names_out(self, mock_sensor_df: pd.DataFrame) -> None:
        extractor = IEQFeatureExtractor()
        extractor.fit(mock_sensor_df)
        names = extractor.get_feature_names_out()
        assert "ieq_index" in names
        assert "thermal_score" in names

    def test_partial_data(self, mock_sensor_df_partial: pd.DataFrame) -> None:
        extractor = IEQFeatureExtractor()
        extractor.fit(mock_sensor_df_partial)
        features = extractor.transform(mock_sensor_df_partial)
        # ieq_index + thermal_score + iaq_score = 3
        assert features.shape[1] == 3

    def test_no_domain_scores(self, mock_sensor_df: pd.DataFrame) -> None:
        extractor = IEQFeatureExtractor(include_domain_scores=False)
        extractor.fit(mock_sensor_df)
        features = extractor.transform(mock_sensor_df)
        assert features.shape[1] == 1  # only ieq_index

    def test_no_ieq_index(self, mock_sensor_df: pd.DataFrame) -> None:
        extractor = IEQFeatureExtractor(include_ieq_index=False)
        extractor.fit(mock_sensor_df)
        features = extractor.transform(mock_sensor_df)
        assert features.shape[1] == 4  # only domain scores

    def test_score_range(self, mock_sensor_df: pd.DataFrame) -> None:
        extractor = IEQFeatureExtractor()
        extractor.fit(mock_sensor_df)
        features = extractor.transform(mock_sensor_df)
        assert np.all(features >= 0.0)
        assert np.all(features <= 100.0)
