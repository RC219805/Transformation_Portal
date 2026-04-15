"""Backwards-compatibility layer for Transformation Portal.

Provides compatibility shims, deprecation warnings, and migration helpers
to ensure existing code continues to work across versions.

This module exports:
    - Decorators: deprecated, moved_to, renamed_function, renamed_class, renamed_module
    - Introspection: get_deprecation_info, is_deprecated
    - Shims: LegacyAPIShim, create_compatibility_wrapper, create_module_alias, DeprecatedConstant
    - Version utilities: Version, parse_version, check_version_compatibility,
                         require_version, version_in_range

Example:
    >>> from transformation_portal.compat import deprecated, renamed_module
    >>>
    >>> @deprecated(replacement="new_function", removal_version="2.0.0")
    ... def old_function():
    ...     return "old"
    ...
    >>> from transformation_portal.compat import Version
    >>> v1 = Version("1.2.3")
    >>> v2 = Version("2.0.0")
    >>> v1 < v2
    True
    >>> v1 in {v1, v2}  # Hashable for sets/dicts
    True
"""

from .decorators import (
    deprecated,
    get_deprecation_info,
    is_deprecated,
    moved_to,
    renamed_class,
    renamed_function,
    renamed_module,
)
from .shims import (
    DeprecatedConstant,
    LegacyAPIShim,
    create_compatibility_wrapper,
    create_module_alias,
)
from .version import (
    Version,
    check_version_compatibility,
    parse_version,
    require_version,
    version_in_range,
)

__all__ = [
    # Decorators
    "deprecated",
    "renamed_function",
    "renamed_class",
    "renamed_module",
    "moved_to",
    "get_deprecation_info",
    "is_deprecated",
    # Shims
    "LegacyAPIShim",
    "create_compatibility_wrapper",
    "create_module_alias",
    "DeprecatedConstant",
    # Version utilities
    "Version",
    "parse_version",
    "check_version_compatibility",
    "require_version",
    "version_in_range",
]

__version__ = "1.1.0"
