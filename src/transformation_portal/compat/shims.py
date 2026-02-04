"""Shim layers for intercepting and redirecting legacy API calls."""

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class LegacyAPIShim:
    """A proxy object that intercepts access to deprecated attributes.

    Useful for maintaining backward compatibility when an entire class structure
    has changed. It acts as a facade, forwarding calls to a new implementation
    while logging warnings.
    """

    def __init__(
        self,
        real_object: Any,
        name: str,
        attribute_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize shim.

        Args:
            real_object: The new object implementation to proxy to.
            name: The name of the old (deprecated) object for logging.
            attribute_map: Dict mapping old_attr_name -> new_attr_name.
        """
        self._real_object = real_object
        self._name = name
        self._attribute_map = attribute_map or {}

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access and forward to real object."""
        # Check if we have a mapping for this attribute
        target_name = self._attribute_map.get(name, name)

        if not hasattr(self._real_object, target_name):
            raise AttributeError(
                f"'{self._name}' (shim for {type(self._real_object).__name__}) "
                f"has no attribute '{name}' (mapped to '{target_name}')"
            )

        # Log usage of deprecated shim
        logger.warning(
            f"Accessing deprecated object '{self._name}'. "
            f"Forwarding '{name}' -> '{target_name}'. "
            f"Please update code to use {type(self._real_object).__name__} directly."
        )

        return getattr(self._real_object, target_name)

    def __repr__(self) -> str:
        return f"<LegacyAPIShim for {self._real_object!r}>"


def create_compatibility_wrapper(func: Callable[..., Any], old_name: str, new_name: str) -> Callable[..., Any]:
    """Create a wrapper function that warns when the old name is used."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.warning(f"'{old_name}' is deprecated. Please use '{new_name}'.")
        return func(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__doc__ = f"Deprecated alias for {new_name}. {func.__doc__}"
    return wrapper
