"""Tests for logging configuration module."""

from __future__ import annotations

import logging

from comfio.logging_config import setup_logging


class TestSetupLogging:
    def test_returns_comfio_logger(self) -> None:
        logger = setup_logging(force=True)
        assert logger.name == "comfio"

    def test_level_string(self) -> None:
        logger = setup_logging(level="DEBUG", force=True)
        assert logger.level == logging.DEBUG

    def test_level_int(self) -> None:
        logger = setup_logging(level=logging.INFO, force=True)
        assert logger.level == logging.INFO

    def test_has_handler(self) -> None:
        logger = setup_logging(force=True)
        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_force_replaces_handlers(self) -> None:
        setup_logging(force=True)
        logger1 = setup_logging(force=True)
        # Should still have exactly one handler after force
        assert len(logger1.handlers) == 1

    def test_no_force_keeps_first_config(self) -> None:
        setup_logging(level="DEBUG", force=True)
        # Second call without force should be a no-op
        setup_logging(level="ERROR")
        logger = logging.getLogger("comfio")
        assert logger.level == logging.DEBUG  # First call wins
