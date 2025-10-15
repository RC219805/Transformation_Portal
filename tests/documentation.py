"""Shared documentation helpers for architectural tests."""

from __future__ import annotations

from datetime import date
from functools import wraps
from collections.abc import Iterable
from typing import Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., object])


def documents(note: str) -> Callable[[F], F]:
    """Annotate a test with the documentation note it enforces."""

    def decorator(func: F) -> F:
        func.__doc__ = note if func.__doc__ is None else f"{note}\n{func.__doc__}"
        return func

    return decorator


def demonstrates(concepts: Iterable[object] | object) -> Callable[[F], F]:
    """Annotate a test subject with the concept(s) it proves out.

    The decorator mirrors :func:`documents` but is semantically distinct: it
    links a test to one or more architectural principles that the test
    exercises.  The relationship is recorded on the decorated object via a
    ``__demonstrates__`` attribute so that meta-tests or documentation tooling
    can surface the mapping between principles and coverage.

    The decorator accepts either a single concept (for convenience) or an
    iterable of concepts.  Concepts may be strings, classes, or any other
    descriptive object; they are stored verbatim and an informative docstring
    note is prefixed so the intent shows up in pytest's verbose output.
    """

    if isinstance(concepts, (str, bytes)) or not isinstance(concepts, Iterable):
        concept_list = [concepts]
    else:
        concept_list = list(concepts)

    def decorator(obj: F) -> F:
        note_parts = []
        for concept in concept_list:
            if hasattr(concept, "__name__"):
                note_parts.append(concept.__name__)  # type: ignore[arg-type]
            else:
                note_parts.append(str(concept))

        note = "Demonstrates: " + ", ".join(note_parts)
        obj.__doc__ = note if obj.__doc__ is None else f"{note}\n{obj.__doc__}"
        setattr(obj, "__demonstrates__", tuple(concept_list))
        return obj

    return decorator


def valid_until(iso_date: str, *, reason: str) -> Callable[[F], F]:
    """Fail the decorated test once the provided date has passed.

    This helper acts as a temporal contract around architectural decisions: once
    the stated expiry date is exceeded the test will raise an assertion failure,
    signalling that the guarded assumption needs to be reviewed.
    """

    deadline = date.fromisoformat(iso_date)

    def decorator(func: F) -> F:
        note = f"Valid until {deadline.isoformat()} – {reason}"

        @wraps(func)
        def wrapper(*args: object, **kwargs: object):
            today = date.today()
            if today > deadline:
                raise AssertionError(
                    "Temporal contract expired: "
                    f"{func.__name__!r} requires review after {deadline.isoformat()} "
                    f"because {reason}."
                )
            return func(*args, **kwargs)

        wrapper.__doc__ = (
            note if func.__doc__ is None else f"{note}\n{func.__doc__}"
        )
        return cast(F, wrapper)

    return decorator


__all__ = ["documents", "demonstrates", "valid_until"]