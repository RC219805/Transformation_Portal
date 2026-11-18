"""Helpers for request values the portal cannot echo back as JSON.

Python's JSON parser accepts ``NaN``, ``Infinity`` and out-of-range literals
such as ``1e400`` (which parse to non-finite floats) as well as ``\\ud800``
escapes (which parse to unpaired surrogates). Neither can be serialised again
by the strict response encoder (``allow_nan=False`` plus UTF-8 encoding), so a
config preview that echoes them crashes with a bare 500. These helpers let the
preview flag such arguments as field errors and echo them as ``null``, which is
the contract the managed photography previews already follow.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

NON_FINITE_NUMBER_REASON = "non_finite_number"
INVALID_UNICODE_REASON = "invalid_unicode_string"
_UNECHOABLE_KEY_FIELD = "args"


def _has_unpaired_surrogate(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def json_payload_rejection_reason(payload: Any) -> Optional[str]:
    """Return a reason code when ``payload`` holds a value JSON cannot echo.

    The walk is iterative so deeply nested payloads cannot exhaust the
    interpreter stack. ``None`` means the payload is safe to echo.
    """

    stack: List[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, float):
            if not math.isfinite(value):
                return NON_FINITE_NUMBER_REASON
        elif isinstance(value, str):
            if _has_unpaired_surrogate(value):
                return INVALID_UNICODE_REASON
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and _has_unpaired_surrogate(key):
                    return INVALID_UNICODE_REASON
                stack.append(item)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return None


def unechoable_arg_fields(args: Any) -> List[Tuple[str, str]]:
    """Return ``(field, reason)`` for each top-level arg JSON cannot echo.

    An argument whose *name* is itself unechoable is reported under the
    generic ``args`` field so the name is never copied into the response.
    """

    if not isinstance(args, dict):
        return []
    issues: List[Tuple[str, str]] = []
    for key, value in args.items():
        if isinstance(key, str) and _has_unpaired_surrogate(key):
            issues.append((_UNECHOABLE_KEY_FIELD, INVALID_UNICODE_REASON))
            continue
        reason = json_payload_rejection_reason(value)
        if reason is not None:
            issues.append((str(key), reason))
    return issues


def json_safe_echo(value: Any) -> Any:
    """Copy ``value`` with unechoable floats and strings replaced by ``None``.

    Dict entries whose key cannot be encoded are dropped. Values that are
    already echoable are returned unchanged (same object).
    """

    if json_payload_rejection_reason(value) is None:
        return value
    if isinstance(value, (float, str)):
        return None
    if isinstance(value, dict):
        return {
            key: json_safe_echo(item)
            for key, item in value.items()
            if not (isinstance(key, str) and _has_unpaired_surrogate(key))
        }
    return [json_safe_echo(item) for item in value]
