"""Custom exception classes for graceful failure handling.

When analyzing months of building data, a failed sensor reading at 2:00 AM
should flag a warning and skip the row, rather than terminating the entire
compilation script.
"""

from __future__ import annotations


class comfioError(Exception):
    """Base exception for all comfio errors."""


class MissingSensorDataError(comfioError):
    """Raised when required sensor columns are absent from the input data."""


class OutOfRangeError(comfioError):
    """Raised when a measured value falls outside physically realistic bounds."""


class InvalidUnitError(comfioError):
    """Raised when a unit conversion is requested for an unknown or mismatched unit."""


class DomainNotAvailableError(comfioError):
    """Raised when a domain score is requested but the domain was not computed."""


class WeightConfigurationError(comfioError):
    """Raised when weighting schema configuration is invalid (e.g., weights don't sum to 1)."""
