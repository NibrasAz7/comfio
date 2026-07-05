"""Tests for Keras IEQ adapter.

Skipped if tensorflow is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from comfio.ml.keras_adapter import IEQPreprocessingLayer


class TestIEQPreprocessingLayer:
    def test_call_with_dataframe(self, mock_sensor_df: pd.DataFrame) -> None:
        import tensorflow as tf

        layer = IEQPreprocessingLayer()
        layer.adapt(mock_sensor_df)
        result = layer(mock_sensor_df)
        assert isinstance(result, (tf.Tensor, np.ndarray))
        assert result.shape[0] == len(mock_sensor_df)

    def test_feature_count(self, mock_sensor_df: pd.DataFrame) -> None:
        layer = IEQPreprocessingLayer()
        layer.adapt(mock_sensor_df)
        result = layer(mock_sensor_df)
        # ieq_index + 4 domain scores = 5
        assert result.shape[1] == 5

    def test_no_domain_scores(self, mock_sensor_df: pd.DataFrame) -> None:
        layer = IEQPreprocessingLayer(include_domain_scores=False)
        layer.adapt(mock_sensor_df)
        result = layer(mock_sensor_df)
        assert result.shape[1] == 1

    def test_score_range(self, mock_sensor_df: pd.DataFrame) -> None:
        layer = IEQPreprocessingLayer()
        layer.adapt(mock_sensor_df)
        result = layer(mock_sensor_df).numpy()
        assert np.all(result >= 0.0)
        assert np.all(result <= 100.0)

    def test_get_config(self) -> None:
        layer = IEQPreprocessingLayer()
        config = layer.get_config()
        assert "include_domain_scores" in config
        assert "thermal_category" in config
