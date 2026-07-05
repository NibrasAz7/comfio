"""scikit-learn-compatible transformers for IEQ feature extraction.

Provides ``TransformerMixin`` classes that extract IEQ features from
sensor DataFrames, making them usable directly in sklearn pipelines.

Requires the ``[ml]`` extra: ``pip install comfio[ml]``
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
from comfio.integration.global_ieq import calculate_global_ieq
from comfio.integration.weights import WeightSchema, default_weights


class IEQFeatureExtractor:
    """scikit-learn transformer that extracts IEQ features from sensor data.

    Wraps comfio domain evaluations into a ``fit/transform`` interface
    suitable for use in ``sklearn.pipeline.Pipeline``.

    Parameters
    ----------
    weights : WeightSchema or None
        Weighting schema for Global IEQ Index. Defaults to default_weights().
    include_domain_scores : bool, default True
        Whether to include per-domain scores in the output features.
    include_ieq_index : bool, default True
        Whether to include the Global IEQ Index in the output features.
    thermal_category : str, default "B"
        ISO 7730 thermal comfort category.
    visual_task_type : str, default "general"
        EN 12464-1 task type for visual evaluation.
    acoustic_nc_level : str, default "NC-35"
        Noise Criteria level for acoustic evaluation.
    iaq_threshold_level : str, default "good"
        CO₂ threshold level for IAQ evaluation.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from comfio.ml.sklearn_transformers import IEQFeatureExtractor
    >>> pipe = Pipeline([
    ...     ("ieq", IEQFeatureExtractor()),
    ...     ("model", RandomForestRegressor()),
    ... ])
    """

    def __init__(
        self,
        weights: WeightSchema | None = None,
        include_domain_scores: bool = True,
        include_ieq_index: bool = True,
        thermal_category: str = "B",
        visual_task_type: str = "general",
        acoustic_nc_level: str = "NC-35",
        iaq_threshold_level: str = "good",
    ) -> None:
        self.weights = weights or default_weights()
        self.include_domain_scores = include_domain_scores
        self.include_ieq_index = include_ieq_index
        self.thermal_category = thermal_category
        self.visual_task_type = visual_task_type
        self.acoustic_nc_level = acoustic_nc_level
        self.iaq_threshold_level = iaq_threshold_level
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> IEQFeatureExtractor:
        """Fit the transformer (no-op, just records feature names).

        Parameters
        ----------
        X : pandas.DataFrame
            Sensor data with columns matching comfio canonical names
            (or aliases that SensorData can auto-detect).
        y : Any
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        IEQFeatureExtractor
            self
        """
        sensor = SensorData(df=X)
        domains = sensor.available_domains()
        names: list[str] = []
        if self.include_ieq_index:
            names.append("ieq_index")
        if self.include_domain_scores:
            names.extend(f"{d}_score" for d in domains)
        self._feature_names = names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform sensor DataFrames into IEQ feature arrays.

        Parameters
        ----------
        X : pandas.DataFrame
            Sensor data.

        Returns
        -------
        np.ndarray
            2-D array of shape (n_samples, n_features) with IEQ features.

        Notes
        -----
        The output feature matrix contains:

        - ``ieq_index`` (if ``include_ieq_index=True``)
        - ``{domain}_score`` for each available domain (if ``include_domain_scores=True``)

        Features are ordered: IEQ index first, then domain scores in
        the order returned by ``SensorData.available_domains()``.
        """
        sensor = SensorData(df=X)
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

        ieq_result = calculate_global_ieq(
            thermal=thermal_res,
            visual=visual_res,
            acoustic=acoustic_res,
            iaq=iaq_res,
            weights=self.weights,
        )

        # Build feature matrix
        features: list[np.ndarray] = []
        if self.include_ieq_index:
            features.append(ieq_result.index.reshape(-1, 1))
        if self.include_domain_scores:
            for d in domains:
                features.append(ieq_result.domain_scores[d].reshape(-1, 1))

        if not features:
            return np.empty((len(X), 0))

        return np.hstack(features)

    def get_feature_names_out(self) -> np.ndarray:
        """Return feature names for sklearn compatibility.

        Returns
        -------
        np.ndarray
            Array of feature name strings.
        """
        return np.array(self._feature_names)
