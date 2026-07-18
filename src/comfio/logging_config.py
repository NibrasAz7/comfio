"""Logging configuration for comfio.

Provides :func:`setup_logging` to configure comfio's logger with a
sensible default format. Individual modules use
``logging.getLogger(__name__)`` to get a logger that inherits this
configuration.

Example
-------
>>> import comfio
>>> comfio.setup_logging(level="INFO")
>>> from comfio import evaluate_thermal
>>> # warnings from evaluation will now appear in stderr
"""

from __future__ import annotations

import logging
import sys
from typing import IO, Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(
    level: LogLevel | int = "WARNING",
    *,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
    stream: IO[str] = sys.stderr,
    force: bool = False,
) -> logging.Logger:
    """Configure comfio's logger.

    Parameters
    ----------
    level : str or int, default "WARNING"
        Logging level. One of "DEBUG", "INFO", "WARNING", "ERROR",
        "CRITICAL", or the corresponding integer constant.
    fmt : str
        Log message format string (passed to ``logging.Formatter``).
    datefmt : str
        Date format string for ``asctime``.
    stream : file-like
        Output stream for log messages (default: ``sys.stderr``).
    force : bool, default False
        If True, reconfigure even if :func:`setup_logging` was already
        called. If False, the first call wins (subsequent calls are
        no-ops unless ``force=True``).

    Returns
    -------
    logging.Logger
        The configured ``comfio`` logger.

    Notes
    -----
    By default, comfio does **not** configure logging — Python's
    "last handler wins" rule means library code should not add handlers
    to the root logger. Call :func:`setup_logging` explicitly in your
    script or notebook to see comfio's log messages.

    Examples
    --------
    >>> import comfio
    >>> logger = comfio.setup_logging(level="INFO")
    >>> logger.name
    'comfio'
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return logging.getLogger("comfio")

    comfio_logger = logging.getLogger("comfio")
    comfio_logger.setLevel(level if isinstance(level, int) else getattr(logging, level))

    # Remove existing handlers to avoid duplicates on re-configuration
    if force:
        for h in comfio_logger.handlers[:]:
            comfio_logger.removeHandler(h)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt, datefmt))
    comfio_logger.addHandler(handler)
    comfio_logger.propagate = False

    _CONFIGURED = True
    return comfio_logger
