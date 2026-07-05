"""Tests for PyTorch IEQ Dataset.

Skipped if torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from comfio.ml.torch_dataset import IEQTimeSeriesDataset


class TestIEQTimeSeriesDataset:
    def test_len(self, mock_sensor_df: pd.DataFrame) -> None:
        dataset = IEQTimeSeriesDataset(mock_sensor_df, window_size=10, stride=1)
        # (100 - 10) // 1 + 1 = 91
        assert len(dataset) == 91

    def test_len_with_stride(self, mock_sensor_df: pd.DataFrame) -> None:
        dataset = IEQTimeSeriesDataset(mock_sensor_df, window_size=10, stride=5)
        # (100 - 10) // 5 + 1 = 19
        assert len(dataset) == 19

    def test_getitem_raw(self, mock_sensor_df: pd.DataFrame) -> None:
        dataset = IEQTimeSeriesDataset(
            mock_sensor_df, window_size=10, include_ieq=False, include_raw=True
        )
        sample = dataset[0]
        assert "raw" in sample
        assert sample["raw"].shape == (10, 9)  # 9 sensor columns

    def test_getitem_ieq(self, mock_sensor_df: pd.DataFrame) -> None:
        dataset = IEQTimeSeriesDataset(
            mock_sensor_df, window_size=10, include_ieq=True, include_raw=False
        )
        sample = dataset[0]
        assert "ieq_index" in sample
        assert sample["ieq_index"].shape == (10,)
        assert "domain_scores" in sample

    def test_getitem_both(self, mock_sensor_df: pd.DataFrame) -> None:
        dataset = IEQTimeSeriesDataset(
            mock_sensor_df, window_size=10, include_ieq=True, include_raw=True
        )
        sample = dataset[0]
        assert "raw" in sample
        assert "ieq_index" in sample

    def test_score_range(self, mock_sensor_df: pd.DataFrame) -> None:
        dataset = IEQTimeSeriesDataset(
            mock_sensor_df, window_size=10, include_ieq=True, include_raw=False
        )
        sample = dataset[0]
        assert np.all(sample["ieq_index"] >= 0.0)
        assert np.all(sample["ieq_index"] <= 100.0)

    def test_raw_feature_names(self, mock_sensor_df: pd.DataFrame) -> None:
        dataset = IEQTimeSeriesDataset(mock_sensor_df, window_size=10)
        names = dataset.raw_feature_names
        assert "air_temp_c" in names
        assert "illuminance_lux" in names

    def test_with_dataloader(self, mock_sensor_df: pd.DataFrame) -> None:
        from torch.utils.data import DataLoader

        dataset = IEQTimeSeriesDataset(
            mock_sensor_df, window_size=10, include_ieq=True, include_raw=False
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["ieq_index"].shape == (4, 10)
