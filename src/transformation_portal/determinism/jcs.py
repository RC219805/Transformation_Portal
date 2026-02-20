"""RFC 8785 (JCS) canonical JSON serialization.

This implementation is intentionally small and self-contained to avoid
dependency drift. It supports the JSON data model used by the determinism
harness evidence artifacts.

Supported types: dict[str, Any], list[Any], str, int, float, bool, None.

Key sorting is performed by UTF-16 code units (per RFC 8785), not by
Unicode code points.
"""

from __future__ import annotations

import json
import math
from typing import Any


def _utf16_sort_key(s: str) -> bytes:
    # UTF-16 big endian yields a byte sequence whose lexicographic order matches
    # UTF-16 code unit order.
    return s.encode("utf-16-be")


def _escape_json_string(s: str) -> str:
    # json.dumps produces a valid JSON string literal, including the surrounding quotes.
    # ensure_ascii=False keeps unicode code points as UTF-8 where legal.
    return json.dumps(s, ensure_ascii=False, separators=(",", ":"))


def _js_number_to_string(x: float) -> str:
    """ECMAScript-like Number.prototype.toString for finite IEEE-754 binary64.

    RFC 8785 requires ECMAScript number serialization. This function aims to
    match JS formatting choices that differ from Python, notably:
      - exponent threshold: use exponential notation for abs(x) < 1e-6 or abs(x) >= 1e21
      - exponent has no leading zeros (e.g., 1e-7 not 1e-07)
      - -0 is serialized as 0
      - integers are serialized without trailing ".0"
    """
    if not math.isfinite(x):
        raise ValueError("Non-finite numbers are not permitted in JCS JSON.")
    if x == 0.0:
        return "0"

    neg = x < 0
    ax = -x if neg else x

    # Python's repr() is a shortest round-trippable representation, which is a good
    # basis for JCS. We then normalize formatting to ECMAScript conventions.
    s = repr(ax)

    # Convert to either decimal or exponential notation per JS magnitude thresholds.
    if 1e-6 <= ax < 1e21:
        # JS chooses decimal notation in this range.
        if "e" in s or "E" in s:
            mant, exp_str = s.lower().split("e", 1)
            exp = int(exp_str)
            if "." in mant:
                int_part, frac_part = mant.split(".", 1)
                digits = int_part + frac_part
                dec_places = len(frac_part)
            else:
                digits = mant
                dec_places = 0
            # Remove leading zeros from digits (mantissa is >=1 so safe).
            digits = digits.lstrip("0") or "0"
            total_exp = exp - dec_places

            if total_exp >= 0:
                out = digits + ("0" * total_exp)
            else:
                pos = len(digits) + total_exp
                if pos > 0:
                    out = digits[:pos] + "." + digits[pos:]
                else:
                    out = "0." + ("0" * (-pos)) + digits

            # Trim trailing zeros after decimal point.
            if "." in out:
                out = out.rstrip("0").rstrip(".")
            s = out
        else:
            # Already decimal, but Python might emit trailing ".0" for ints.
            if s.endswith(".0"):
                s = s[:-2]
    else:
        # JS chooses exponential notation outside [1e-6, 1e21).
        if "e" not in s and "E" not in s:
            # Convert decimal to scientific by using Python's repr in scientific form.
            # format(..., 'e') is not shortest, so we instead re-use repr() by forcing
            # through Decimal exponent extraction is overkill; instead, rely on repr()
            # which already uses scientific for very small/large numbers. For mid-range,
            # we should not land here because threshold check would have used decimal.
            s = "{:e}".format(ax)
        s = s.lower()
        mant, exp_str = s.split("e", 1)
        exp = int(exp_str)

        # Normalize mantissa: remove trailing zeros and optional '.'.
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")

        # Normalize exponent: remove leading zeros.
        sign = "-" if exp < 0 else "+"
        exp_abs = abs(exp)
        exp_digits = str(exp_abs)
        # JS omits '+' for positive exponent? Actually JS uses 'e+NN' for positive.
        s = mant + "e" + sign + exp_digits

    if neg:
        s = "-" + s
    return s


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _escape_json_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _js_number_to_string(value)
    if isinstance(value, list):
        return "[" + ",".join(_serialize(v) for v in value) + "]"
    if isinstance(value, dict):
        # Keys must be strings in JSON. Sort by UTF-16 code units.
        items = []
        for k in sorted(value.keys(), key=_utf16_sort_key):
            if not isinstance(k, str):
                raise TypeError("JCS only supports string object keys.")
            items.append(_escape_json_string(k) + ":" + _serialize(value[k]))
        return "{" + ",".join(items) + "}"
    raise TypeError(f"Unsupported type for JCS serialization: {type(value)!r}")


def dumps(value: Any) -> str:
    """Return canonical JSON string (UTF-8 safe) per RFC 8785."""
    return _serialize(value)


def dumpb(value: Any) -> bytes:
    """Return canonical JSON bytes (UTF-8) per RFC 8785."""
    return dumps(value).encode("utf-8")


def sha256_hex_of_canonical_json(value: Any) -> str:
    import hashlib

    return hashlib.sha256(dumpb(value)).hexdigest()
