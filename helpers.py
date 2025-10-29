# helpers.py
"""Shared architectural helpers and Golden Hour Courtyard utilities."""

from __future__ import annotations
from datetime import date
from functools import wraps
from typing import Any, Callable, Sequence, Type, Union

# -------------------------
# documents decorator
# -------------------------


def documents(note: str) -> Callable[[Callable], Callable]:
    """Attach a docstring note to a function, preserving existing doc."""
    def decorator(func: Callable) -> Callable:
        if func.__doc__:
            func.__doc__ = f"{note}\n{func.__doc__}"
        else:
            func.__doc__ = note
        return func
    return decorator

# -------------------------
# demonstrates decorator
# -------------------------


ConceptType = Union[str, int, Type]


def demonstrates(concepts: Union[ConceptType, Sequence[ConceptType]]) -> Callable[[Callable], Callable]:
    """Annotate a function as demonstrating one or more concepts."""
    if not isinstance(concepts, (list, tuple)):
        concepts_seq = (concepts,)
    else:
        concepts_seq = tuple(concepts)

    def decorator(func: Callable) -> Callable:
        func.__demonstrates__ = concepts_seq
        # Build a docstring prefix
        parts = []
        for c in concepts_seq:
            if hasattr(c, "__name__"):
                parts.append(c.__name__)
            else:
                parts.append(str(c))
        prefix = "Demonstrates: " + ", ".join(parts)
        if func.__doc__:
            func.__doc__ = f"{prefix}\n{func.__doc__}"
        else:
            func.__doc__ = prefix
        return func
    return decorator

# -------------------------
# valid_until decorator
# -------------------------


def valid_until(expiration: str, reason: str = "expired") -> Callable[[Callable], Callable]:
    """Ensure the decorated function can only be executed until a given ISO date."""
    expiration_date = date.fromisoformat(expiration)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            today = date.today()
            if today > expiration_date:
                raise AssertionError(f"Function {func.__name__} is no longer valid: {reason}")
            return func(*args, **kwargs)

        wrapper.__doc__ = f"Valid until {expiration} — {reason}"
        return wrapper

    return decorator
