"""Decorators for handling API deprecations and migrations.

This module provides decorators to mark functions, methods, and classes as
deprecated while maintaining backward compatibility. All decorators emit
DeprecationWarning when deprecated code is accessed and update docstrings
for IDE/Sphinx support.

Example:
    >>> from transformation_portal.compat.decorators import deprecated
    >>>
    >>> @deprecated(replacement="new_function", removal_version="2.0.0")
    ... def old_function():
    ...     return "result"
    ...
    >>> old_function()  # Emits DeprecationWarning
    'result'
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Optional, Type, TypeVar

T = TypeVar("T", bound=Callable[..., Any])
C = TypeVar("C", bound=Type[Any])

# Track deprecation metadata for introspection
_DEPRECATION_REGISTRY: dict[str, dict[str, Any]] = {}


def deprecated(
    replacement: Optional[str] = None,
    removal_version: Optional[str] = None,
    reason: Optional[str] = None,
    *,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 2,
) -> Callable[[T], T]:
    """Mark a function or method as deprecated.

    Emits a DeprecationWarning when called and updates the docstring.
    The decorated function's metadata is preserved via functools.wraps.

    Args:
        replacement: Name of the function to use instead.
        removal_version: Version when this will be strictly removed.
        reason: Custom explanation for deprecation.
        category: Warning category to emit (default: DeprecationWarning).
                  Use FutureWarning for end-user visible deprecations.
        stacklevel: Stack level for the warning (default: 2).

    Returns:
        Decorator that wraps the function with deprecation behavior.

    Example:
        >>> @deprecated(replacement="new_func", removal_version="2.0.0")
        ... def old_func(x: int) -> int:
        ...     return x * 2
        ...
        >>> old_func(5)  # Emits DeprecationWarning
        10
    """

    def decorator(func: T) -> T:
        # Build deprecation message
        parts = [f"{func.__name__} is deprecated."]
        if replacement:
            parts.append(f"Use '{replacement}' instead.")
        if removal_version:
            parts.append(f"Scheduled for removal in v{removal_version}.")
        if reason:
            parts.append(f"Reason: {reason}")
        message = " ".join(parts)

        # Register deprecation metadata for introspection
        func_key = f"{func.__module__}.{func.__qualname__}"
        _DEPRECATION_REGISTRY[func_key] = {
            "replacement": replacement,
            "removal_version": removal_version,
            "reason": reason,
            "message": message,
        }

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        # Update docstring for Sphinx/IDE support
        doc = wrapper.__doc__ or ""
        wrapper.__doc__ = f".. warning:: DEPRECATED\n   {message}\n\n{doc}"

        # Attach deprecation metadata to wrapper for runtime introspection
        wrapper._deprecated_info = _DEPRECATION_REGISTRY[func_key]  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def get_deprecation_info(func: Callable[..., Any]) -> Optional[dict[str, Any]]:
    """Get deprecation metadata for a function, if it was decorated with @deprecated.

    Args:
        func: The function to check.

    Returns:
        Dictionary with deprecation info, or None if not deprecated.

    Example:
        >>> @deprecated(replacement="new_func")
        ... def old_func(): pass
        ...
        >>> info = get_deprecation_info(old_func)
        >>> info['replacement']
        'new_func'
    """
    return getattr(func, "_deprecated_info", None)


def is_deprecated(func: Callable[..., Any]) -> bool:
    """Check if a function is marked as deprecated.

    Args:
        func: The function to check.

    Returns:
        True if the function has been decorated with @deprecated.
    """
    return hasattr(func, "_deprecated_info")


def moved_to(new_location: str, *, removal_version: Optional[str] = None) -> Callable[[T], T]:
    """Indicate a function has moved to a new module/namespace.

    Args:
        new_location: The new fully-qualified path (e.g., "module.submodule.function").
        removal_version: Version when the old location will be removed.

    Returns:
        Decorator that marks the function as moved.

    Example:
        >>> @moved_to("transformation_portal.new_module.new_func")
        ... def old_func():
        ...     pass
    """
    return deprecated(
        replacement=new_location,
        removal_version=removal_version,
        reason="Moved to new namespace.",
    )


def renamed_function(new_name: str, *, removal_version: Optional[str] = None) -> Callable[[T], T]:
    """Indicate a function has been renamed.

    Args:
        new_name: The new function name.
        removal_version: Version when the old name will be removed.

    Returns:
        Decorator that marks the function as renamed.

    Example:
        >>> @renamed_function("calculate_total")
        ... def calc_total():
        ...     pass
    """
    return deprecated(
        replacement=new_name,
        removal_version=removal_version,
        reason="Function renamed.",
    )


def renamed_class(
    new_name: str,
    *,
    removal_version: Optional[str] = None,
    category: type[Warning] = DeprecationWarning,
) -> Callable[[C], C]:
    """Indicate a class has been renamed.

    Emits a DeprecationWarning when the class is instantiated.

    Args:
        new_name: The new class name.
        removal_version: Version when the old name will be removed.
        category: Warning category to emit (default: DeprecationWarning).

    Returns:
        Class decorator that adds deprecation warning on instantiation.

    Example:
        >>> @renamed_class("NewProcessor", removal_version="2.0.0")
        ... class OldProcessor:
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> obj = OldProcessor(42)  # Emits DeprecationWarning
    """

    def decorator(cls: C) -> C:
        message = f"Class {cls.__name__} is deprecated. Use {new_name} instead."
        if removal_version:
            message += f" Scheduled for removal in v{removal_version}."

        # Register deprecation metadata
        cls_key = f"{cls.__module__}.{cls.__qualname__}"
        _DEPRECATION_REGISTRY[cls_key] = {
            "replacement": new_name,
            "removal_version": removal_version,
            "message": message,
        }

        # Hook __init__ to warn on instantiation
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            warnings.warn(message, category, stacklevel=2)
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init  # type: ignore[method-assign]

        # Attach deprecation info to class
        cls._deprecated_info = _DEPRECATION_REGISTRY[cls_key]  # type: ignore[attr-defined]

        return cls

    return decorator


def renamed_module(old_name: str, new_name: str, *, stacklevel: int = 3) -> None:
    """Emit a warning for a renamed module (call at top of old module).

    Should be called at module level immediately after imports in the
    deprecated module to alert users importing from the old location.

    Args:
        old_name: The old module name being deprecated.
        new_name: The new module name to use instead.
        stacklevel: Stack level for the warning (default: 3 for module-level call).

    Example:
        In deprecated_module.py::

            # At top of file, after imports
            from transformation_portal.compat import renamed_module
            renamed_module(__name__, "transformation_portal.new_module")

            # Rest of module for backward compatibility...
    """
    warnings.warn(
        f"Module {old_name} is deprecated. Import from {new_name} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
