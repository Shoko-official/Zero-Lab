"""Logging setup for local runs."""

from __future__ import annotations

import logging
import sys

from zero_lab.core.config import RuntimeConfig

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(config: RuntimeConfig) -> logging.Logger:
    config.run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("zero_lab")
    logger.handlers.clear()
    logger.setLevel(config.log_level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(config.log_level)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(config.run_dir / "zero_lab.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(config.log_level)
    logger.addHandler(file_handler)

    return logger
