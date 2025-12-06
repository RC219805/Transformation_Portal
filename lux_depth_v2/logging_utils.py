from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", json_logs: bool = False) -> logging.Logger:
    """Configure a process-wide logger suitable for batch or service mode."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("lux_depth_v2")
    logger.setLevel(lvl)

    # avoid duplicate handlers (e.g., in notebooks or reload)
    if logger.handlers:
        return logger

    h = logging.StreamHandler(sys.stdout)
    h.setLevel(lvl)

    if json_logs:
        # minimal JSON logger, good for structured ingestion
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                import json
                payload = {
                    "level": record.levelname,
                    "name": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc_info"] = self.formatException(record.exc_info)
                return json.dumps(payload, ensure_ascii=False)

        h.setFormatter(JsonFormatter())
    else:
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.addHandler(h)
    logger.propagate = False
    return logger
