"""TensorFlow/Keras preprocessing layer adapters for IEQ data.

Provides a ``tf.keras.layers.Layer`` that wraps comfio IEQ calculations
for use in Keras preprocessing pipelines.

Requires the ``[keras]`` extra: ``pip install comfio[keras]``
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


class IEQPreprocessingLayer:
    """Keras-compatible preprocessing layer for IEQ feature extraction.

    Wraps comfio domain evaluations into a Keras layer interface
    suitable for use in ``tf.keras`` preprocessing pipelines.

    Parameters
    ----------
    weights : WeightSchema or None
        Weighting schema for IEQ Index. Defaults to default_weights().
    include_domain_scores : bool, default True
        Whether to include per-domain scores in output features.
    include_ieq_index : bool, default True
        Whether to include the Global IEQ Index in output features.
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
    >>> import tensorflow as tf  # doctest: +SKIP
    >>> from comfio.ml.keras_adapter import IEQPreprocessingLayer  # doctest: +SKIP
    >>> inputs = tf.keras.Input(shape=(8,), name="sensor_data")  # doctest: +SKIP
    >>> ieq_layer = IEQPreprocessingLayer()  # doctest: +SKIP
    >>> features = ieq_layer(inputs)  # doctest: +SKIP
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
        # Lazy import to avoid hard dependency on tensorflow
        import tensorflow as tf

        self.weights = weights or default_weights()
        self.include_domain_scores = include_domain_scores
        self.include_ieq_index = include_ieq_index
        self.thermal_category = thermal_category
        self.visual_task_type = visual_task_type
        self.acoustic_nc_level = acoustic_nc_level
        self.iaq_threshold_level = iaq_threshold_level
        self._tf = tf

        # Build the actual Keras layer
        self._layer = tf.keras.layers.Layer(name="ieq_preprocessing")

    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute IEQ features from a DataFrame."""
        sensor = SensorData(df=df)
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

        features: list[np.ndarray] = []
        if self.include_ieq_index:
            features.append(ieq_result.index.reshape(-1, 1))
        if self.include_domain_scores:
            for d in domains:
                features.append(ieq_result.domain_scores[d].reshape(-1, 1))

        if not features:
            return np.empty((len(df), 0))
        return np.hstack(features)

    def __call__(self, inputs: Any) -> Any:
        """Apply IEQ preprocessing to inputs.

        Accepts either a pandas DataFrame (eager computation) or a
        TensorFlow tensor (deferred via custom op).

        Parameters
        ----------
        inputs : pandas.DataFrame or tf.Tensor
            Sensor data input.

        Returns
        -------
        tf.Tensor or np.ndarray
            IEQ feature array.
        """
        if isinstance(inputs, pd.DataFrame):
            features = self._compute_features(inputs)
            return self._tf.constant(features, dtype=self._tf.float32)

        # For tf.Tensor inputs, use a py_function wrapper
        tf = self._tf

        def eager_fn(arr: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(arr, columns=self._column_names)
            return self._compute_features(df)

        return tf.py_function(eager_fn, [inputs], tf.float32)

    def adapt(self, data: pd.DataFrame) -> None:
        """Adapt the layer to training data (records feature names).

        Parameters
        ----------
        data : pandas.DataFrame
            Training sensor data.
        """
        sensor = SensorData(df=data)
        self._column_names = list(sensor.column_map.keys())
        domains = sensor.available_domains()
        names: list[str] = []
        if self.include_ieq_index:
            names.append("ieq_index")
        if self.include_domain_scores:
            names.extend(f"{d}_score" for d in domains)
        self._feature_names = names

    def get_config(self) -> dict[str, Any]:
        """Return layer configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return {
            "include_domain_scores": self.include_domain_scores,
            "include_ieq_index": self.include_ieq_index,
            "thermal_category": self.thermal_category,
            "visual_task_type": self.visual_task_type,
            "acoustic_nc_level": self.acoustic_nc_level,
            "iaq_threshold_level": self.iaq_threshold_level,
        }
