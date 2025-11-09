"""Compatibility shims for legacy APIs."""

import functools
from typing import Any, Callable, Dict, Optional


class LegacyAPIShim:
    """Base class for creating compatibility shims for legacy APIs.

    Example:
        >>> class OldProcessorShim(LegacyAPIShim):
        ...     def __init__(self):
        ...         super().__init__(
        ...             "OldProcessor",
        ...             "transformation_portal.processors.NewProcessor"
        ...         )
        ...
        ...     def _get_implementation(self):
        ...         from transformation_portal.processors import NewProcessor
        ...         return NewProcessor
    """

    def __init__(self, old_name: str, new_location: str, removal_version: Optional[str] = None):
        """Initialize shim.

        Args:
            old_name: Name of old API
            new_location: Import path of new API
            removal_version: Version when shim will be removed
        """
        self._old_name = old_name
        self._new_location = new_location
        self._removal_version = removal_version
        self._warned = False

    def _show_warning(self):
        """Show deprecation warning (only once)."""
        if not self._warned:
            import warnings
            msg = f"'{self._old_name}' is deprecated. Import from '{self._new_location}' instead"
            if self._removal_version:
                msg += f". Will be removed in version {self._removal_version}"
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
            self._warned = True

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to new implementation."""
        self._show_warning()
        impl = self._get_implementation()
        return getattr(impl, name)

    def __call__(self, *args, **kwargs) -> Any:
        """Delegate calls to new implementation."""
        self._show_warning()
        impl = self._get_implementation()
        return impl(*args, **kwargs)

    def _get_implementation(self) -> Any:
        """Get the new implementation (must be overridden)."""
        raise NotImplementedError("Subclasses must implement _get_implementation")


def create_compatibility_wrapper(
    old_func: Callable,
    new_func: Callable,
    param_mapping: Optional[Dict[str, str]] = None,
    removal_version: Optional[str] = None
) -> Callable:
    """Create a compatibility wrapper that maps old parameters to new ones.

    Args:
        old_func: Original function (for metadata)
        new_func: New function to wrap
        param_mapping: Mapping of old parameter names to new ones
        removal_version: Version when wrapper will be removed

    Returns:
        Wrapped function with parameter mapping

    Example:
        >>> def new_process(image_path, output_dir, quality=95):
        ...     pass
        ...
        >>> def old_process(input_file, output_folder, jpg_quality=95):
        ...     pass
        ...
        >>> compat_process = create_compatibility_wrapper(
        ...     old_process,
        ...     new_process,
        ...     param_mapping={
        ...         'input_file': 'image_path',
        ...         'output_folder': 'output_dir',
        ...         'jpg_quality': 'quality'
        ...     },
        ...     removal_version="2.0.0"
        ... )
    """
    param_mapping = param_mapping or {}

    @functools.wraps(old_func)
    def wrapper(*args, **kwargs):
        import warnings

        # Show deprecation warning
        msg = f"'{old_func.__name__}' is deprecated. Use '{new_func.__name__}' instead"
        if removal_version:
            msg += f". Will be removed in version {removal_version}"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        # Map old parameter names to new ones
        new_kwargs = {}
        for old_name, value in kwargs.items():
            new_name = param_mapping.get(old_name, old_name)
            new_kwargs[new_name] = value

        return new_func(*args, **new_kwargs)

    return wrapper


def create_alias(original: Any, alias_name: str, removal_version: Optional[str] = None) -> Any:
    """Create a deprecated alias for a function or class.

    Args:
        original: Original function/class
        alias_name: Name of the alias
        removal_version: Version when alias will be removed

    Returns:
        Aliased function/class with deprecation warning

    Example:
        >>> def new_function():
        ...     pass
        ...
        >>> old_function = create_alias(new_function, "old_function", "2.0.0")
    """
    if isinstance(original, type):
        # For classes
        class AliasedClass(original):
            def __init__(self, *args, **kwargs):
                import warnings
                msg = f"'{alias_name}' is deprecated. Use '{original.__name__}' instead"
                if removal_version:
                    msg += f". Will be removed in version {removal_version}"
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                super().__init__(*args, **kwargs)

        AliasedClass.__name__ = alias_name
        return AliasedClass
    else:
        # For functions
        @functools.wraps(original)
        def aliased_function(*args, **kwargs):
            import warnings
            msg = f"'{alias_name}' is deprecated. Use '{original.__name__}' instead"
            if removal_version:
                msg += f". Will be removed in version {removal_version}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return original(*args, **kwargs)

        aliased_function.__name__ = alias_name
        return aliased_function
