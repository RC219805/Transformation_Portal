"""Lightweight log classification helpers.

Two related concerns the diagnosis flagged in the runtime logs:

1. Routine ``GET /healthz`` access lines drown out actionable failures.
   :func:`install_healthcheck_log_filter` attaches a ``logging.Filter`` to the
   uvicorn access logger that drops successful healthcheck lines when the env
   var ``TP_LOG_HEALTHCHECKS`` is ``0``. Errors are *never* dropped.

2. Existing logs are unstructured prose, so triage requires regex grepping.
   :func:`log_kind` emits a key=value line tagged with a ``kind`` field
   (``healthcheck``, ``telemetry``, ``protected_api``, ``sse``, ``artifact``,
   ``auth``, ``audit``). The function deliberately uses ``%``-style formatting
   instead of ``json.dumps`` so it can live outside the JSON-serialization
   governance allowlist.

Both helpers are additive; importing this module does not change behaviour.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable

_HEALTHCHECK_PATHS = ("/healthz", "/ready")
_LOGGER_NAME = "uvicorn.access"
_FILTER_ATTR = "_tp_healthcheck_filter_installed"

# Allowed `kind` tags. The list is small on purpose so log triage stays
# narrow — adding a new kind is a deliberate change.
_VALID_KINDS = frozenset(
    {
        "healthcheck",
        "telemetry",
        "protected_api",
        "sse",
        "artifact",
        "auth",
        "audit",
    }
)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip() not in {"0", "false", "False", ""}


class _SuccessfulHealthcheckFilter(logging.Filter):
    """Drop ``GET /healthz`` and ``GET /ready`` access lines with status < 400."""

    def __init__(self, paths: Iterable[str] = _HEALTHCHECK_PATHS) -> None:
        super().__init__()
        # Match the uvicorn access default format:
        #   '%(client_addr)s - "%(request_line)s" %(status_code)s'
        # We do not want to over-engineer the parser — falling back to keeping
        # the line on any uncertainty preserves visibility.
        path_alt = "|".join(re.escape(p.rstrip("/")) for p in paths)
        self._pattern = re.compile(r'"\s*(?:GET|HEAD)\s+(?:' + path_alt + r')(?:[?\s][^"]*)?"\s+(\d{3})')

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - logging API
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        match = self._pattern.search(message)
        if match is None:
            return True
        try:
            status = int(match.group(1))
        except ValueError:  # pragma: no cover - regex guarantees \d{3}
            return True
        # Errors must always be visible, even for healthchecks.
        return status >= 400


def install_healthcheck_log_filter(
    logger_name: str = _LOGGER_NAME,
    *,
    env_var: str = "TP_LOG_HEALTHCHECKS",
) -> bool:
    """Install (or remove) the healthcheck access-log filter.

    Returns ``True`` when the filter is now installed, ``False`` otherwise.

    Behaviour:
        * ``TP_LOG_HEALTHCHECKS=1`` (default) keeps every access line; the
          filter is removed if it was previously installed.
        * ``TP_LOG_HEALTHCHECKS=0`` installs the suppression filter exactly
          once; subsequent calls are idempotent.
    """

    logger = logging.getLogger(logger_name)
    if _env_flag(env_var):
        existing = getattr(logger, _FILTER_ATTR, None)
        if existing is not None:
            logger.removeFilter(existing)
            setattr(logger, _FILTER_ATTR, None)
        return False
    if getattr(logger, _FILTER_ATTR, None) is None:
        f = _SuccessfulHealthcheckFilter()
        logger.addFilter(f)
        setattr(logger, _FILTER_ATTR, f)
    return True


def log_kind(
    logger: logging.Logger,
    kind: str,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured key=value log line tagged with ``kind``.

    The output format is ``kind=<kind> event=<event> k1=v1 k2=v2 ...``;
    values are passed through ``str()`` then any embedded whitespace is
    collapsed so the line stays single-line and grep-friendly.
    """

    if kind not in _VALID_KINDS:
        kind = "audit"
    parts = [f"kind={kind}", f"event={event}"]
    for key, value in fields.items():
        text = str(value)
        text = re.sub(r"\s+", " ", text).strip()
        parts.append(f"{key}={text}")
    logger.log(level, " ".join(parts))


__all__ = ["install_healthcheck_log_filter", "log_kind"]
