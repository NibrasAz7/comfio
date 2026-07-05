"""Static and dynamic weighting schemas for Global IEQ Index calculation.

The weighting schema defines how much each domain (thermal, visual,
acoustic, IAQ) contributes to the overall IEQ score. Weights must sum
to 1.0 when all four domains are present. Missing domains are handled
by renormalizing the remaining weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from comfio.core.exceptions import WeightConfigurationError

DomainName = Literal["thermal", "visual", "acoustic", "iaq"]

# Default weights based on multi-domain IEQ studies.
# Source: Pierson et al. (2019), Cao et al. (2012), Frontczak & Wargocki (2011)
# Thermal comfort is consistently rated most important, followed by IAQ,
# then visual and acoustic.
DEFAULT_WEIGHTS: dict[str, float] = {
    "thermal": 0.40,
    "iaq": 0.25,
    "visual": 0.20,
    "acoustic": 0.15,
}

# Alternative weight presets from literature
WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "default": DEFAULT_WEIGHTS,
    # Equal weighting
    "equal": {"thermal": 0.25, "iaq": 0.25, "visual": 0.25, "acoustic": 0.25},
    # School children study (Yang et al. 2020)
    "school": {"thermal": 0.27, "iaq": 0.26, "visual": 0.24, "acoustic": 0.23},
    # Office workers (emphasis on thermal and IAQ)
    "office": {"thermal": 0.45, "iaq": 0.30, "visual": 0.15, "acoustic": 0.10},
    # Healthcare (emphasis on IAQ)
    "healthcare": {"thermal": 0.25, "iaq": 0.40, "visual": 0.15, "acoustic": 0.20},
}

WeightPreset = Literal["default", "equal", "school", "office", "healthcare"]


@dataclass
class WeightSchema:
    """Weighting schema for multi-domain IEQ index calculation.

    Attributes
    ----------
    weights : dict[str, float]
        Mapping from domain name to weight (0-1).
    preset_name : str
        Name of the preset used (or "custom").
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    preset_name: str = "default"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that weights are non-negative and sum to ~1.0."""
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=0.01):
            raise WeightConfigurationError(
                f"Weights must sum to 1.0 (got {total:.4f}). "
                f"Weights: {self.weights}"
            )
        for kiwi, w in self.weights.items():
            if w < 0 or w > 1:
                raise WeightConfigurationError(
                    f"Weight for '{kiwi}' must be between 0 and 1 (got {w})."
                )

    def get_normalized(self, available_domains: list[str]) -> dict[str, float]:
        """Return weights renormalized for the available domains.

        If a domain is missing (e.g., no acoustic sensor), its weight is
        redistributed proportionally across the remaining domains.

        Parameters
        ----------
        available_domains : list[str]
            Domains that have data (subset of the weight keys).

        Returns
        -------
        dict[str, float]
            Normalized weights for the available domains only.
        """
        present = {d: self.weights.get(d, 0.0) for d in available_domains if d in self.weights}
        total = sum(present.values())
        if total <= 0:
            # Equal weighting fallback
            n = len(available_domains)
            return {d: 1.0 / n for d in available_domains}
        return {d: w / total for d, w in present.items()}

    def get_array(self, available_domains: list[str]) -> np.ndarray:
        """Return normalized weights as a numpy array in domain order.

        Parameters
        ----------
        available_domains : list[str]
            Domains that have data.

        Returns
        -------
        np.ndarray
            1-D array of weights, same order as ``available_domains``.
        """
        normed = self.get_normalized(available_domains)
        return np.array([normed[d] for d in available_domains], dtype=float)


def default_weights() -> WeightSchema:
    """Return the default weighting schema.

    Returns
    -------
    WeightSchema
        Schema with default weights (thermal=0.40, iaq=0.25, visual=0.20, acoustic=0.15).
    """
    return WeightSchema(weights=dict(DEFAULT_WEIGHTS), preset_name="default")


def preset_weights(preset: WeightPreset) -> WeightSchema:
    """Return a named preset weighting schema.

    Parameters
    ----------
    preset : str
        One of "default", "equal", "school", "office", "healthcare".

    Returns
    -------
    WeightSchema
        Schema with the preset weights.
    """
    if preset not in WEIGHT_PRESETS:
        raise WeightConfigurationError(
            f"Unknown preset '{preset}'. Available: {list(WEIGHT_PRESETS.keys())}"
        )
    return WeightSchema(weights=dict(WEIGHT_PRESETS[preset]), preset_name=preset)


def custom_weights(thermal: float, visual: float, acoustic: float, iaq: float) -> WeightSchema:
    """Create a custom weighting schema.

    Parameters
    ----------
    thermal : float
        Thermal domain weight (0-1).
    visual : float
        Visual domain weight (0-1).
    acoustic : float
        Acoustic domain weight (0-1).
    iaq : float
        IAQ domain weight (0-1).

    Returns
    -------
    WeightSchema
        Schema with the custom weights.
    """
    return WeightSchema(
        weights={"thermal": thermal, "visual": visual, "acoustic": acoustic, "iaq": iaq},
        preset_name="custom",
    )
