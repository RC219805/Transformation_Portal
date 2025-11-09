"""Version checking and compatibility utilities."""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

from packaging import version as pkg_version


@dataclass
class Version:
    """Represents a semantic version."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None

    @classmethod
    def from_string(cls, version_str: str) -> 'Version':
        """Parse version string.

        Args:
            version_str: Version string (e.g., "1.2.3" or "1.2.3-beta")

        Returns:
            Version instance
        """
        parsed = pkg_version.parse(version_str)

        if isinstance(parsed, pkg_version.Version):
            return cls(
                major=parsed.major,
                minor=parsed.minor,
                patch=parsed.micro,
                prerelease=str(parsed.pre) if parsed.pre else None
            )
        else:
            raise ValueError(f"Invalid version string: {version_str}")

    def __str__(self) -> str:
        """String representation."""
        version_str = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version_str += f"-{self.prerelease}"
        return version_str

    def __lt__(self, other: 'Version') -> bool:
        """Less than comparison."""
        return pkg_version.parse(str(self)) < pkg_version.parse(str(other))

    def __le__(self, other: 'Version') -> bool:
        """Less than or equal comparison."""
        return pkg_version.parse(str(self)) <= pkg_version.parse(str(other))

    def __gt__(self, other: 'Version') -> bool:
        """Greater than comparison."""
        return pkg_version.parse(str(self)) > pkg_version.parse(str(other))

    def __ge__(self, other: 'Version') -> bool:
        """Greater than or equal comparison."""
        return pkg_version.parse(str(self)) >= pkg_version.parse(str(other))

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, Version):
            return False
        return pkg_version.parse(str(self)) == pkg_version.parse(str(other))


def check_version_compatibility(
    current_version: str,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Check if current version is compatible with requirements.

    Args:
        current_version: Current version string
        min_version: Minimum required version
        max_version: Maximum supported version

    Returns:
        Tuple of (is_compatible, error_message)

    Example:
        >>> is_compatible, msg = check_version_compatibility("1.5.0", "1.0.0", "2.0.0")
        >>> assert is_compatible
    """
    current = pkg_version.parse(current_version)

    if min_version:
        minimum = pkg_version.parse(min_version)
        if current < minimum:
            return False, f"Version {current_version} is below minimum required {min_version}"

    if max_version:
        maximum = pkg_version.parse(max_version)
        if current > maximum:
            return False, f"Version {current_version} exceeds maximum supported {max_version}"

    return True, None


def require_version(
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
):
    """Decorator to enforce version requirements.

    Args:
        min_version: Minimum required version
        max_version: Maximum supported version

    Example:
        >>> @require_version(min_version="1.0.0", max_version="2.0.0")
        ... def my_function():
        ...     pass
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from transformation_portal import __version__

            is_compatible, error_msg = check_version_compatibility(
                __version__, min_version, max_version
            )

            if not is_compatible:
                raise RuntimeError(
                    f"Version incompatibility in {func.__name__}: {error_msg}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_portal_version() -> str:
    """Get current Transformation Portal version.

    Returns:
        Version string
    """
    try:
        from transformation_portal import __version__
        return __version__
    except ImportError:
        return "0.0.0"


def is_version_at_least(required_version: str) -> bool:
    """Check if portal version is at least the required version.

    Args:
        required_version: Required version string

    Returns:
        True if current version >= required version
    """
    current = get_portal_version()
    is_compatible, _ = check_version_compatibility(
        current, min_version=required_version
    )
    return is_compatible
