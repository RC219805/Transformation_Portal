"""SQL safety helpers for metrics queries."""

from __future__ import annotations

from typing import Optional

SQL_LIMIT_MIN = 1
SQL_LIMIT_MAX = 1000


def normalize_query_limit(limit: Optional[int]) -> Optional[int]:
    """Validate and bound SQL LIMIT values.

    Args:
        limit: Optional query limit.

    Returns:
        None if limit is None, otherwise an integer in range 1..1000.

    Raises:
        ValueError: If limit is not an integer or is less than 1.
    """
    if limit is None:
        return None

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"limit must be an integer in range {SQL_LIMIT_MIN}..{SQL_LIMIT_MAX}")

    if limit < SQL_LIMIT_MIN:
        raise ValueError(f"limit must be >= {SQL_LIMIT_MIN}")

    return min(limit, SQL_LIMIT_MAX)
