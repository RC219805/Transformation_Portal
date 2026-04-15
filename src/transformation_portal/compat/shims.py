"""Shim layers for intercepting and redirecting legacy API calls.

This module provides proxy objects and wrapper functions for maintaining
backward compatibility when API structures change. Shims act as facades,
forwarding calls to new implementations while logging warnings to help
users migrate.

Example:
    >>> from transformation_portal.compat.shims import LegacyAPIShim
    >>>
    >>> class NewAPI:
    ...     def new_method(self):
    ...         return "result"
    ...
    >>> shim = LegacyAPIShim(NewAPI(), "OldAPI", {"old_method": "new_method"})
    >>> shim.old_method()  # Logs warning, returns "result"
    'result'
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Track warned attributes to avoid log spam
_warned_attributes: Set[str] = set()


class LegacyAPIShim:
    """A proxy object that intercepts access to deprecated attributes.

    Useful for maintaining backward compatibility when an entire class structure
    has changed. It acts as a facade, forwarding calls to a new implementation
    while logging warnings.

    The shim supports:
    - Attribute access with optional name mapping
    - Method calls (forwarded transparently)
    - Configurable warning behavior (once per attribute vs always)

    Attributes:
        _real_object: The underlying new implementation being proxied.
        _name: Name of the deprecated object for logging.
        _attribute_map: Mapping from old attribute names to new ones.

    Example:
        >>> class NewProcessor:
        ...     def process(self, data):
        ...         return f"processed: {data}"
        ...
        >>> shim = LegacyAPIShim(
        ...     NewProcessor(),
        ...     "OldProcessor",
        ...     {"run": "process"}  # old_name -> new_name
        ... )
        >>> shim.run("test")  # Logs warning, calls process()
        'processed: test'
    """

    __slots__ = ("_real_object", "_name", "_attribute_map", "_warn_once", "_warned")

    def __init__(
        self,
        real_object: Any,
        name: str,
        attribute_map: Optional[Dict[str, str]] = None,
        *,
        warn_once: bool = False,
    ) -> None:
        """Initialize shim.

        Args:
            real_object: The new object implementation to proxy to.
            name: The name of the old (deprecated) object for logging.
            attribute_map: Dict mapping old_attr_name -> new_attr_name.
            warn_once: If True, only warn once per attribute name (reduces log noise).
        """
        object.__setattr__(self, "_real_object", real_object)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_attribute_map", attribute_map or {})
        object.__setattr__(self, "_warn_once", warn_once)
        object.__setattr__(self, "_warned", set())

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access and forward to real object."""
        real_object = object.__getattribute__(self, "_real_object")
        attr_map = object.__getattribute__(self, "_attribute_map")
        obj_name = object.__getattribute__(self, "_name")
        warn_once = object.__getattribute__(self, "_warn_once")
        warned = object.__getattribute__(self, "_warned")

        # Check if we have a mapping for this attribute
        target_name = attr_map.get(name, name)

        if not hasattr(real_object, target_name):
            raise AttributeError(
                f"'{obj_name}' (shim for {type(real_object).__name__}) "
                f"has no attribute '{name}'" + (f" (mapped to '{target_name}')" if name != target_name else "")
            )

        # Log usage of deprecated shim (optionally once per attribute)
        warn_key = f"{obj_name}.{name}"
        should_warn = not warn_once or warn_key not in warned

        if should_warn:
            if warn_once:
                warned.add(warn_key)
            logger.warning(
                "Accessing deprecated object '%s'. Forwarding '%s' -> '%s'. " "Please update code to use %s directly.",
                obj_name,
                name,
                target_name,
                type(real_object).__name__,
            )

        return getattr(real_object, target_name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute setting to real object."""
        real_object = object.__getattribute__(self, "_real_object")
        attr_map = object.__getattribute__(self, "_attribute_map")
        target_name = attr_map.get(name, name)
        setattr(real_object, target_name, value)

    def __repr__(self) -> str:
        """Return repr for the shim."""
        real_object = object.__getattribute__(self, "_real_object")
        return f"<LegacyAPIShim for {real_object!r}>"

    def __str__(self) -> str:
        """Return str for the shim."""
        real_object = object.__getattribute__(self, "_real_object")
        obj_name = object.__getattribute__(self, "_name")
        return f"<{obj_name} (deprecated, use {type(real_object).__name__})>"

    def _get_real_object(self) -> Any:
        """Get the underlying real object (for testing/debugging)."""
        return object.__getattribute__(self, "_real_object")


def create_compatibility_wrapper(
    func: F,
    old_name: str,
    new_name: str,
    *,
    warn_once: bool = True,
) -> F:
    """Create a wrapper function that warns when the old name is used.

    Args:
        func: The new function to wrap.
        old_name: The old/deprecated function name.
        new_name: The new function name.
        warn_once: If True, only warn once per wrapper (default: True).

    Returns:
        Wrapped function that logs deprecation warnings.

    Example:
        >>> def new_process(data):
        ...     return f"processed: {data}"
        ...
        >>> old_process = create_compatibility_wrapper(
        ...     new_process, "old_process", "new_process"
        ... )
        >>> old_process("test")  # Logs warning on first call
        'processed: test'
    """
    warned = False

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal warned
        if not warn_once or not warned:
            logger.warning(
                "'%s' is deprecated. Please use '%s' instead.",
                old_name,
                new_name,
            )
            warned = True
        return func(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__doc__ = f"Deprecated alias for {new_name}.\n\n{func.__doc__ or ''}"
    return wrapper  # type: ignore[return-value]


def create_module_alias(
    real_module: Any,
    deprecated_name: str,
    *,
    attribute_map: Optional[Dict[str, str]] = None,
    warn_once: bool = True,
) -> LegacyAPIShim:
    """Create a module-level shim for a deprecated module name.

    This is useful when a module has been renamed but you want to maintain
    backward compatibility for imports.

    Args:
        real_module: The new module object.
        deprecated_name: The old module name for warnings.
        attribute_map: Optional mapping of old to new attribute names.
        warn_once: If True, only warn once per attribute access.

    Returns:
        LegacyAPIShim that can be assigned to sys.modules[old_name].

    Example:
        In old_module/__init__.py::

            import sys
            from transformation_portal.compat.shims import create_module_alias
            from transformation_portal import new_module

            sys.modules[__name__] = create_module_alias(
                new_module, __name__, warn_once=True
            )
    """
    return LegacyAPIShim(
        real_module,
        deprecated_name,
        attribute_map=attribute_map,
        warn_once=warn_once,
    )


class DeprecatedConstant:
    """A descriptor that warns when a deprecated constant is accessed.

    Use this to mark module-level constants as deprecated while keeping
    them functional.

    Example:
        >>> class Constants:
        ...     OLD_VALUE = DeprecatedConstant(42, "OLD_VALUE", "NEW_VALUE")
        ...     NEW_VALUE = 42
        ...
        >>> Constants.OLD_VALUE  # Logs warning, returns 42
        42
    """

    __slots__ = ("_value", "_old_name", "_new_name", "_warned")

    def __init__(self, value: Any, old_name: str, new_name: str) -> None:
        """Initialize deprecated constant.

        Args:
            value: The constant value.
            old_name: The deprecated name.
            new_name: The new name to use instead.
        """
        self._value = value
        self._old_name = old_name
        self._new_name = new_name
        self._warned = False

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        """Return value and warn on first access."""
        if not self._warned:
            logger.warning(
                "Constant '%s' is deprecated. Use '%s' instead.",
                self._old_name,
                self._new_name,
            )
            self._warned = True
        return self._value

    def __repr__(self) -> str:
        """Return repr for the deprecated constant."""
        return f"DeprecatedConstant({self._value!r}, {self._old_name!r}, {self._new_name!r})"
