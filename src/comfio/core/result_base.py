"""Base mixin for all comfio result dataclasses.

Provides ``to_dict()``, ``to_json()``, and ``to_dataframe()`` convenience
methods so users don't need to know about ``dataclasses.asdict()`` or
manually extract fields one by one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from typing import Any, cast

import numpy as np


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types not natively serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ResultBase:
    """Mixin providing serialization helpers for result dataclasses.

    All comfio ``@dataclass`` result types inherit from this class to gain
    ``to_dict()``, ``to_json()``, and ``to_dataframe()`` methods.

    Examples
    --------
    >>> from comfio import evaluate_thermal
    >>> import numpy as np
    >>> result = evaluate_thermal(
    ...     tdb=np.array([24.0]), tr=np.array([24.0]),
    ...     vr=np.array([0.1]), rh=np.array([50.0]),
    ...     met=1.2, clo=0.5,
    ... )
    >>> d = result.to_dict()  # dict with pmv, ppd, compliant, category
    >>> j = result.to_json()  # JSON string
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a plain dictionary.

        Numpy arrays are kept as-is (use ``to_json()`` for JSON-safe output).

        Returns
        -------
        dict
            Dictionary mapping field names to values.
        """
        return cast(dict[str, Any], asdict(self))  # type: ignore[call-overload]

    def to_json(self, indent: int = 2) -> str:
        """Serialize the result to a JSON string.

        Numpy arrays are converted to lists, and numpy scalars to Python
        primitives, so the output is valid JSON.

        Parameters
        ----------
        indent : int, default 2
            JSON indentation level.

        Returns
        -------
        str
            JSON string representation of the result.
        """
        return json.dumps(asdict(self), indent=indent, default=_json_default)  # type: ignore[call-overload]

    def to_dataframe(self) -> Any:
        """Convert array-valued fields to a pandas DataFrame.

        Only fields containing numpy arrays (or lists of equal length) are
        included as columns. Scalar fields (e.g. ``category: str``) are
        broadcast to all rows.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per timestamp and one column per array field.

        Raises
        ------
        ValueError
            If no array-valued fields are present.
        """
        import pandas as pd

        data: dict[str, Any] = {}
        scalar_data: dict[str, Any] = {}

        for f in fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if isinstance(val, np.ndarray) and val.ndim == 1:
                data[f.name] = val
            elif isinstance(val, list) and len(val) > 0 and not isinstance(val[0], (str, dict)):
                data[f.name] = np.asarray(val)
            else:
                scalar_data[f.name] = val

        if not data:
            raise ValueError(
                f"{type(self).__name__} has no 1-D array fields; "
                "to_dataframe() requires at least one array-valued field."
            )

        df = pd.DataFrame(data)

        # Broadcast scalar fields to all rows
        for k, v in scalar_data.items():
            df[k] = v

        return df


def is_result_instance(obj: Any) -> bool:
    """Check if an object is a dataclass instance that inherits ResultBase."""
    return is_dataclass(obj) and not isinstance(obj, type) and isinstance(obj, ResultBase)
