"""Decorators for handling API deprecations and migrations."""

import functools
import inspect
import warnings
from typing import Any, Callable, Optional, Type, TypeVar

T = TypeVar("T", bound=Callable[..., Any])
C = TypeVar("C", bound=Type[Any])


def deprecated(
    replacement: Optional[str] = None,
    removal_version: Optional[str] = None,
    reason: Optional[str] = None,
) -> Callable[[T], T]:
    """Mark a function or method as deprecated.

    Emits a DeprecationWarning when called and updates the docstring.

    Args:
        replacement: Name of the function to use instead.
        removal_version: Version when this will be strictly removed.
        reason: Custom explanation for deprecation.
    """

    def decorator(func: T) -> T:
        message = f"{func.__name__} is deprecated."
        if replacement:
            message += f" Use '{replacement}' instead."
        if removal_version:
            message += f" Scheduled for removal in v{removal_version}."
        if reason:
            message += f" Reason: {reason}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Update docstring for Sphinx/IDE support
        doc = wrapper.__doc__ or ""
        wrapper.__doc__ = (
            f".. warning:: DEPRECATED\n   {message}\n\n{doc}"
        )
        return wrapper  # type: ignore

    return decorator


def moved_to(new_location: str) -> Callable[[T], T]:
    """Indicate a function has moved to a new module/namespace."""
    return deprecated(replacement=new_location, reason="Moved to new namespace.")


def renamed_function(new_name: str) -> Callable[[T], T]:
    """Indicate a function has been renamed."""
    return deprecated(replacement=new_name, reason="Function renamed.")


def renamed_class(new_name: str) -> Callable[[C], C]:
    """Indicate a class has been renamed."""
    
    def decorator(cls: C) -> C:
        # Hook __init__ to warn on instantiation
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                f"Class {cls.__name__} is deprecated. Use {new_name} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init # type: ignore
        return cls

    return decorator


def renamed_module(old_name: str, new_name: str) -> None:
    """Emit a warning for a renamed module (call at top of old module)."""
    warnings.warn(
        f"Module {old_name} is deprecated. Import from {new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
