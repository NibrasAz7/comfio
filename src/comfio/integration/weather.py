"""Weather data integration for adaptive comfort models.

Provides :func:`fetch_outdoor_temperature` and :func:`fetch_prevailing_temp`
to retrieve historical outdoor air temperature from the `meteostat
<https://dev.meteostat.net/python/>`_ database, which aggregates
observations from national weather services (NOAA, DWD, etc.).

These helpers feed directly into :func:`comfio.evaluate_adaptive_ashrae`
(``t_prevail`` parameter) and :func:`comfio.evaluate_adaptive_en`
(``t_running_mean`` parameter).

.. note::

    This module requires the ``meteostat`` package (included in comfio's
    core dependencies since v0.1.6). Network access is required — calls
    to ``meteostat`` fetch data from remote APIs. Results are cached
    locally under ``~/.cache/comfio/weather/`` to minimize repeated
    network calls.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default cache directory
_CACHE_DIR = Path.home() / ".cache" / "comfio" / "weather"


def _cache_key(lat: float, lon: float, start: date, end: date) -> Path:
    """Build a cache file path for the given location and date range."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{lat:.4f}_{lon:.4f}_{start.isoformat()}_{end.isoformat()}.parquet"


def fetch_outdoor_temperature(
    lat: float,
    lon: float,
    start: date | datetime,
    end: date | datetime,
    *,
    elevation: float | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch hourly outdoor weather data for a location and date range.

    Uses `meteostat <https://dev.meteostat.net/python/>`_ to retrieve
    historical observations from nearby weather stations, with
    interpolation to the target point.

    Parameters
    ----------
    lat : float
        Latitude of the target location.
    lon : float
        Longitude of the target location.
    start : datetime.date or datetime.datetime
        Start of the date range.
    end : datetime.date or datetime.datetime
        End of the date range.
    elevation : float, optional
        Elevation in meters. If None, meteostat estimates from nearby
        stations.
    use_cache : bool, default True
        If True, check for a cached parquet file before making a network
        request. The cache is stored at
        ``~/.cache/comfio/weather/<lat>_<lon>_<start>_<end>.parquet``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a ``DatetimeIndex`` and columns including
        ``temp`` (air temperature in °C), ``rhum`` (relative humidity %),
        and others depending on station availability.

    Raises
    ------
    ImportError
        If ``meteostat`` is not installed (should not happen since v0.1.6
        — it's a core dependency).
    ValueError
        If no weather data is available for the given location/range.

    Notes
    -----
    This function makes **network requests** to the meteostat data API.
    For reproducibility in tests or CI, mock ``meteostat`` or use
    pre-cached data.

    Examples
    --------
    >>> from datetime import date
    >>> df = fetch_outdoor_temperature(
    ...     lat=50.11, lon=8.68,
    ...     start=date(2025, 1, 1), end=date(2025, 1, 7),
    ... )  # doctest: +SKIP
    >>> "temp" in df.columns  # doctest: +SKIP
    True
    """
    import meteostat

    # Normalise to date objects
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()

    # Check cache
    cache_path = _cache_key(lat, lon, start, end)
    if use_cache and cache_path.exists():
        logger.info("Loading cached weather data from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching weather data for (%.4f, %.4f) %s→%s", lat, lon, start, end)

    # Create a Point — meteostat 2.x converts it to a virtual station
    point = meteostat.Point(lat, lon, int(elevation) if elevation is not None else None)

    # Fetch hourly data via the functional API (meteostat 2.x)
    ts = meteostat.hourly(point, start, end)
    df = ts.fetch()

    if df is None or df.empty:
        raise ValueError(
            f"No weather data available for ({lat}, {lon}) "
            f"from {start} to {end}. Check the location and date range."
        )

    # Cache the result
    if use_cache:
        df.to_parquet(cache_path)
        logger.debug("Cached weather data to %s", cache_path)

    return df


def fetch_prevailing_temp(
    lat: float,
    lon: float,
    end_date: date | datetime,
    *,
    days: int = 7,
    elevation: float | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Fetch the prevailing mean outdoor temperature for adaptive comfort.

    Computes the prevailing mean outdoor temperature as the running mean
    of daily average outdoor temperatures over the preceding ``days``
    days. This value feeds directly into
    :func:`comfio.evaluate_adaptive_ashrae` as the ``t_prevail`` parameter.

    For ASHRAE 55-2023, the default is a 7-day prevailing mean (the
    "prevailing mean outdoor temperature" is the arithmetic mean of the
    daily mean outdoor temperatures over the previous 7 consecutive days).

    Parameters
    ----------
    lat : float
        Latitude of the building location.
    lon : float
        Longitude of the building location.
    end_date : datetime.date or datetime.datetime
        The reference date (typically the last day of the evaluation
        period). The function fetches data for ``end_date - days`` to
        ``end_date``.
    days : int, default 7
        Number of preceding days to average (ASHRAE 55 uses 7; EN 16798-1
        uses a weighted running mean — see :func:`fetch_running_mean`).
    elevation : float, optional
        Elevation in meters.
    use_cache : bool, default True
        Whether to use the local parquet cache.

    Returns
    -------
    np.ndarray
        Scalar array containing the prevailing mean outdoor temperature
        in °C.

    Notes
    -----
    The prevailing mean outdoor temperature for ASHRAE 55 is:

    .. math::

        \\bar{t}_{prevail} = \\frac{1}{n} \\sum_{i=1}^{n} \\bar{t}_{out,i}

    where :math:`\\bar{t}_{out,i}` is the daily mean outdoor temperature
    and *n* is the number of days (default 7).

    Examples
    --------
    >>> from datetime import date
    >>> t_prevail = fetch_prevailing_temp(
    ...     lat=50.11, lon=8.68,
    ...     end_date=date(2025, 6, 30),
    ...     days=7,
    ... )  # doctest: +SKIP
    >>> t_prevail.shape  # doctest: +SKIP
    ()
    """
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    from datetime import timedelta

    start_date = end_date - timedelta(days=days)

    df = fetch_outdoor_temperature(
        lat=lat,
        lon=lon,
        start=start_date,
        end=end_date,
        elevation=elevation,
        use_cache=use_cache,
    )

    # Daily mean temperature
    daily_mean = df["temp"].resample("1D").mean()

    prevailing = float(daily_mean.mean())
    logger.info("Prevailing mean outdoor temp (%d days): %.1f °C", days, prevailing)

    return np.array(prevailing)


def fetch_running_mean(
    lat: float,
    lon: float,
    end_date: date | datetime,
    *,
    days: int = 7,
    alpha: float | None = None,
    elevation: float | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Fetch the running mean outdoor temperature for EN 16798-1.

    Computes the running mean outdoor temperature as defined by
    EN 16798-1:2019 Annex A. This is an exponentially weighted moving
    average of daily mean outdoor temperatures:

    .. math::

        t_{rm} = (1 - \\alpha) \\left( t_{ed-1} + \\alpha t_{ed-2}
            + \\alpha^2 t_{ed-3} + \\cdots \\right)

    where :math:`t_{ed-1}` is yesterday's daily mean, :math:`t_{ed-2}` is
    the day before, etc.

    Parameters
    ----------
    lat : float
        Latitude of the building location.
    lon : float
        Longitude of the building location.
    end_date : datetime.date or datetime.datetime
        The reference date.
    days : int, default 7
        Number of preceding days to include in the weighted average.
    alpha : float, optional
        Weighting factor (0 < alpha < 1). If None, EN 16798-1 recommends
        ``alpha = 0.8`` for typical buildings.
    elevation : float, optional
        Elevation in meters.
    use_cache : bool, default True
        Whether to use the local parquet cache.

    Returns
    -------
    np.ndarray
        Scalar array containing the running mean outdoor temperature
        in °C.

    Notes
    -----
    The EN 16798-1 running mean differs from the ASHRAE 55 prevailing
    mean: the ASHRAE version is a simple arithmetic mean, while the EN
    version uses exponential weighting (recent days matter more).

    Examples
    --------
    >>> from datetime import date
    >>> t_rm = fetch_running_mean(
    ...     lat=50.11, lon=8.68,
    ...     end_date=date(2025, 6, 30),
    ... )  # doctest: +SKIP
    >>> t_rm.shape  # doctest: +SKIP
    ()
    """
    if alpha is None:
        alpha = 0.8  # EN 16798-1 default

    if isinstance(end_date, datetime):
        end_date = end_date.date()

    from datetime import timedelta

    start_date = end_date - timedelta(days=days)

    df = fetch_outdoor_temperature(
        lat=lat,
        lon=lon,
        start=start_date,
        end=end_date,
        elevation=elevation,
        use_cache=use_cache,
    )

    # Daily mean temperatures (most recent first)
    daily_means = df["temp"].resample("1D").mean().dropna().values[::-1]

    if len(daily_means) == 0:
        raise ValueError("No daily mean temperatures available for the given range.")

    # Exponentially weighted running mean
    weights = np.array([(1 - alpha) * alpha**i for i in range(len(daily_means))])
    running_mean = float(np.sum(weights * daily_means[: len(weights)]) / np.sum(weights))

    logger.info("Running mean outdoor temp (α=%.2f, %d days): %.1f °C", alpha, days, running_mean)

    return np.array(running_mean)
