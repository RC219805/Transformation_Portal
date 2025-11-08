"""Backwards-compatibility layer for Transformation Portal.

Provides compatibility shims, deprecation warnings, and migration helpers
to ensure existing code continues to work across versions.

Example:
    >>> from transformation_portal.compat import deprecated, renamed_module
    >>> 
    >>> @deprecated(replacement="new_function", removal_version="2.0.0")
    ... def old_function():
    ...     return "old"
"""

from .decorators import (
    deprecated,
    moved_to,
    renamed_class,
    renamed_function,
    renamed_module,
)
from .shims import (
    LegacyAPIShim,
    create_compatibility_wrapper,
)
from .version import (
    Version,
    check_version_compatibility,
    require_version,
)

__all__ = [
    'deprecated',
    'renamed_function',
    'renamed_class',
    'renamed_module',
    'moved_to',
    'LegacyAPIShim',
    'create_compatibility_wrapper',
    'Version',
    'check_version_compatibility',
    'require_version',
]

__version__ = '1.0.0'
