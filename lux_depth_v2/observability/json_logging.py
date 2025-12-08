from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from .context import get_request_id

# These are standard LogRecord attributes we won't treat as "extra fields"
_RESERVED: Set[str] = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(obj: Any) -> Any:
    # Ensure JSON serialization doesn't explode on odd objects
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Attach request_id to every log record (if present)
        rid = get_request_id()
        setattr(record, "request_id", rid)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": _utc_now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        # Add "extra" fields (anything not reserved)
        for k, v in record.__dict__.items():
            if k in _RESERVED:
                continue
            if k in base:
                continue
            base[k] = _safe(v)

        # Exception info, if any
        if record.exc_info:
            base["exc_type"] = getattr(record.exc_info[0], "__name__", str(record.exc_info[0]))
            base["exc"] = "".join(traceback.format_exception(*record.exc_info)).strip()

        return json.dumps(base, ensure_ascii=False)


def configure_structured_logging(
    *,
    level: Optional[str] = None,
    force_json: Optional[bool] = None,
) -> None:
    """
    Configure root logging.

    - If force_json is True: JSON logs
    - If force_json is False: leave format alone (text)
    - If force_json is None: choose based on LUX_LOG_FORMAT (default json)
    """
    env_fmt = os.getenv("LUX_LOG_FORMAT", "json").strip().lower()
    use_json = (env_fmt == "json") if force_json is None else force_json

    lvl = (level or os.getenv("LUX_LOG_LEVEL", "INFO")).upper()

    root = logging.getLogger()
    root.setLevel(lvl)

    # Ensure a handler exists
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        root.addHandler(handler)

    for h in root.handlers:
        # Attach filter always; formatter depends on JSON mode
        h.addFilter(RequestIdFilter())
        if use_json:
            h.setFormatter(JsonFormatter())

    # Quiet noisy loggers (can expand later)
    logging.getLogger("uvicorn.error").setLevel(lvl)
    logging.getLogger("uvicorn.access").setLevel(lvl)
