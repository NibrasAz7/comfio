"""PyTorch Dataset wrapper for time-series IEQ data.

Provides a ``torch.utils.data.Dataset`` that wraps comfio sensor data
for use with PyTorch ``DataLoader`` in deep learning pipelines.

Requires the ``[torch]`` extra: ``pip install comfio[torch]``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from comfio.core.data_handler import SensorData
from comfio.domains.acoustic import evaluate_acoustic
from comfio.domains.iaq import evaluate_iaq
from comfio.domains.thermal import evaluate_thermal
from comfio.domains.visual import evaluate_visual
from comfio.integration.global_ieq import GlobalIEQResult, calculate_global_ieq
from comfio.integration.weights import WeightSchema, default_weights


class IEQTimeSeriesDataset:
    """PyTorch Dataset for time-series IEQ sensor data.

    Wraps a pandas DataFrame of sensor readings and provides
    windowed samples suitable for sequence models (LSTM, Transformer, etc.).

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor data with columns matching comfio canonical names.
    window_size : int
        Number of timesteps per sample window.
    stride : int, default 1
        Step between consecutive windows.
    weights : WeightSchema or None
        Weighting schema for IEQ Index. Defaults to default_weights().
    include_raw : bool, default True
        Whether to include raw sensor values in each sample.
    include_ieq : bool, default True
        Whether to include computed IEQ scores in each sample.
    thermal_category : str, default "B"
        ISO 7730 thermal comfort category.
    visual_task_type : str, default "general"
        EN 12464-1 task type.
    acoustic_nc_level : str, default "NC-35"
        Noise Criteria level.
    iaq_threshold_level : str, default "good"
        CO₂ threshold level.

    Examples
    --------
    >>> from torch.utils.data import DataLoader  # doctest: +SKIP
    >>> from comfio.ml.torch_dataset import IEQTimeSeriesDataset  # doctest: +SKIP
    >>> dataset = IEQTimeSeriesDataset(df, window_size=24, stride=1)  # doctest: +SKIP
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)  # doctest: +SKIP
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        stride: int = 1,
        weights: WeightSchema | None = None,
        include_raw: bool = True,
        include_ieq: bool = True,
        thermal_category: str = "B",
        visual_task_type: str = "general",
        acoustic_nc_level: str = "NC-35",
        iaq_threshold_level: str = "good",
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.weights = weights or default_weights()
        self.include_raw = include_raw
        self.include_ieq = include_ieq
        self.thermal_category = thermal_category
        self.visual_task_type = visual_task_type
        self.acoustic_nc_level = acoustic_nc_level
        self.iaq_threshold_level = iaq_threshold_level

        # Process the full dataset
        sensor = SensorData(df=df)
        sensor.validate()
        self._sensor = sensor

        # Compute IEQ for the full series
        self._ieq_result = self._compute_ieq(sensor)

        # Build raw feature matrix from validated columns
        self._raw_columns = list(sensor.column_map.keys())
        self._raw_data = np.column_stack(
            [sensor.get_validated(col) for col in self._raw_columns]
        ) if self._raw_columns else np.empty((len(df), 0))

        # Calculate number of windows
        n = len(df)
        self._n_windows = max(0, (n - window_size) // stride + 1)

    def _compute_ieq(self, sensor: SensorData) -> GlobalIEQResult:
        """Compute IEQ scores for the full sensor data."""
        domains = sensor.available_domains()
        thermal_res = None
        visual_res = None
        acoustic_res = None
        iaq_res = None

        if "thermal" in domains:
            thermal_res = evaluate_thermal(
                tdb=sensor.get_validated("air_temp_c"),
                tr=sensor.get_validated("radiant_temp_c"),
                vr=sensor.get_validated("air_velocity_ms"),
                rh=sensor.get_validated("relative_humidity_pct"),
                met=sensor.get_validated("metabolic_rate_met")
                if "metabolic_rate_met" in sensor.column_map
                else 1.2,
                clo=sensor.get_validated("clothing_insulation_clo")
                if "clothing_insulation_clo" in sensor.column_map
                else 0.5,
                category=self.thermal_category,
            )

        if "visual" in domains:
            visual_res = evaluate_visual(
                illuminance=sensor.get_validated("illuminance_lux"),
                task_type=self.visual_task_type,
            )

        if "acoustic" in domains:
            acoustic_res = evaluate_acoustic(
                laeq=sensor.get_validated("noise_laeq_db"),
                nc_level=self.acoustic_nc_level,
            )

        if "iaq" in domains:
            iaq_res = evaluate_iaq(
                co2=sensor.get_validated("co2_ppm"),
                threshold_level=self.iaq_threshold_level,
            )

        return calculate_global_ieq(
            thermal=thermal_res,
            visual=visual_res,
            acoustic=acoustic_res,
            iaq=iaq_res,
            weights=self.weights,
        )

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single windowed sample.

        Parameters
        ----------
        idx : int
            Window index.

        Returns
        -------
        dict
            Dictionary with keys:
            - "raw": (window_size, n_raw_features) array if include_raw
            - "ieq_index": (window_size,) array if include_ieq
            - "domain_scores": dict of (window_size,) arrays if include_ieq
        """
        start = idx * self.stride
        end = start + self.window_size

        sample: dict[str, Any] = {}

        if self.include_raw:
            sample["raw"] = self._raw_data[start:end]

        if self.include_ieq:
            sample["ieq_index"] = self._ieq_result.index[start:end]
            sample["domain_scores"] = {
                d: self._ieq_result.domain_scores[d][start:end]
                for d in self._ieq_result.domains
            }

        return sample

    @property
    def raw_feature_names(self) -> list[str]:
        """Return the names of raw sensor features.

        Returns
        -------
        list[str]
            List of canonical column names used as raw features.
        """
        return list(self._raw_columns)
