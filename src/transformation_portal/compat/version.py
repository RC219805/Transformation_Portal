"""Version checking and compatibility utilities.

This module provides semantic version parsing and comparison for managing
compatibility requirements. It supports standard SemVer format plus common
extensions like prerelease suffixes.

Example:
    >>> from transformation_portal.compat.version import Version, check_version_compatibility
    >>>
    >>> v1 = Version("1.2.3")
    >>> v2 = Version("2.0.0-beta")
    >>> v1 < v2
    True
    >>> check_version_compatibility("1.5.0", "1.0.0")
    True
"""

from __future__ import annotations

import re
from functools import total_ordering
from typing import Optional, Tuple


@total_ordering
class Version:
    """Simple semantic version parser and comparator.

    Supports versions in the format: MAJOR.MINOR[.PATCH][-PRERELEASE]

    The class implements all comparison operators and is hashable,
    making it suitable for use in sets and as dictionary keys.

    Attributes:
        raw: The original version string.
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number (defaults to 0 if not specified).
        prerelease: Prerelease suffix (e.g., "alpha", "beta", "rc1").

    Example:
        >>> v = Version("2.1.0-beta")
        >>> v.major, v.minor, v.patch
        (2, 1, 0)
        >>> v.prerelease
        'beta'
        >>> str(v)
        '2.1.0-beta'
    """

    # Regex for SemVer-ish strings
    _VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:[.-](.+))?$")

    __slots__ = ("raw", "major", "minor", "patch", "prerelease", "_comparison_key")

    def __init__(self, version_str: str) -> None:
        """Initialize Version from a version string.

        Args:
            version_str: Version string (e.g., "1.2.3", "2.0.0-beta").

        Raises:
            ValueError: If version_str is not a valid version format.
        """
        self.raw = version_str
        self.major, self.minor, self.patch, self.prerelease = self._parse(version_str)
        self._comparison_key = self._build_comparison_key()

    def _parse(self, v: str) -> Tuple[int, int, int, str]:
        """Parse version string into components.

        Args:
            v: Version string to parse.

        Returns:
            Tuple of (major, minor, patch, prerelease).

        Raises:
            ValueError: If version string is invalid.
        """
        match = self._VERSION_PATTERN.match(v)
        if not match:
            raise ValueError(f"Invalid version string: {v!r}. " "Expected format: MAJOR.MINOR[.PATCH][-PRERELEASE]")

        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3) or 0)
        prerelease = match.group(4) or ""

        return major, minor, patch, prerelease

    def _build_comparison_key(
        self,
    ) -> tuple[int, int, int, int, tuple[tuple[int, int | str], ...]]:
        """Build a comparison key that follows SemVer prerelease precedence."""
        prerelease_rank = 1
        prerelease_key: tuple[tuple[int, int | str], ...] = ()

        if self.prerelease:
            prerelease_rank = 0
            prerelease_key = tuple(self._encode_prerelease_identifier(identifier) for identifier in self.prerelease.split("."))

        return (
            self.major,
            self.minor,
            self.patch,
            prerelease_rank,
            prerelease_key,
        )

    @staticmethod
    def _encode_prerelease_identifier(identifier: str) -> tuple[int, int | str]:
        """Encode a prerelease identifier for tuple comparison."""
        if identifier.isdigit():
            return (0, int(identifier))
        return (1, identifier)

    @property
    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version."""
        return bool(self.prerelease)

    @property
    def base_version(self) -> str:
        """Get the base version without prerelease suffix."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        """Return repr string for Version."""
        return f"Version({self.raw!r})"

    def __str__(self) -> str:
        """Return the normalized version string."""
        if self.prerelease:
            return f"{self.major}.{self.minor}.{self.patch}-{self.prerelease}"
        return f"{self.major}.{self.minor}.{self.patch}"

    def __hash__(self) -> int:
        """Return hash for Version (enables use in sets/dicts)."""
        return hash(self._comparison_key)

    def __eq__(self, other: object) -> bool:
        """Check equality with another Version."""
        if not isinstance(other, Version):
            return NotImplemented
        return self._comparison_key == other._comparison_key

    def __lt__(self, other: object) -> bool:
        """Check if this version is less than another."""
        if not isinstance(other, Version):
            return NotImplemented
        return self._comparison_key < other._comparison_key

    @classmethod
    def from_tuple(cls, version_tuple: Tuple[int, int, int], prerelease: str = "") -> "Version":
        """Create a Version from a tuple of (major, minor, patch).

        Args:
            version_tuple: Tuple of (major, minor, patch) integers.
            prerelease: Optional prerelease suffix.

        Returns:
            New Version instance.

        Example:
            >>> v = Version.from_tuple((1, 2, 3), "beta")
            >>> str(v)
            '1.2.3-beta'
        """
        major, minor, patch = version_tuple
        if prerelease:
            version_str = f"{major}.{minor}.{patch}-{prerelease}"
        else:
            version_str = f"{major}.{minor}.{patch}"
        return cls(version_str)

    def bump_major(self) -> "Version":
        """Return a new Version with incremented major version.

        Minor and patch are reset to 0, prerelease is cleared.
        """
        return Version.from_tuple((self.major + 1, 0, 0))

    def bump_minor(self) -> "Version":
        """Return a new Version with incremented minor version.

        Patch is reset to 0, prerelease is cleared.
        """
        return Version.from_tuple((self.major, self.minor + 1, 0))

    def bump_patch(self) -> "Version":
        """Return a new Version with incremented patch version.

        Prerelease is cleared.
        """
        return Version.from_tuple((self.major, self.minor, self.patch + 1))


def parse_version(version_str: str) -> Optional[Version]:
    """Parse a version string, returning None on failure instead of raising.

    This is a lenient parser for cases where version validity is uncertain.

    Args:
        version_str: Version string to parse.

    Returns:
        Version instance if valid, None otherwise.

    Example:
        >>> parse_version("1.2.3")
        Version('1.2.3')
        >>> parse_version("invalid") is None
        True
    """
    try:
        return Version(version_str)
    except ValueError:
        return None


def check_version_compatibility(current_version: str, required_version: str) -> bool:
    """Check if current version meets required version.

    Args:
        current_version: The version of the library currently running.
        required_version: The minimum version required.

    Returns:
        True if current >= required.
    """
    try:
        return Version(current_version) >= Version(required_version)
    except ValueError:
        return False


def require_version(min_version: str, *, package_name: str = "Transformation Portal") -> None:
    """Raise RuntimeError if package version is too old.

    Useful for ensuring plugins define a minimum required version of
    Transformation Portal.

    Args:
        min_version: Minimum required version string.
        package_name: Name of the package for error messages.

    Raises:
        RuntimeError: If the current version is below min_version.

    Example:
        >>> require_version("1.0.0")  # Raises if running < 1.0.0
    """
    # Import here to avoid circular dependency
    from transformation_portal import __version__ as current_ver

    if not check_version_compatibility(current_ver, min_version):
        raise RuntimeError(f"{package_name} v{min_version}+ required. Found v{current_ver}.")


def version_in_range(
    version: str,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None,
    *,
    inclusive_max: bool = False,
) -> bool:
    """Check if a version falls within a specified range.

    Args:
        version: Version string to check.
        min_version: Minimum version (inclusive). None means no lower bound.
        max_version: Maximum version. None means no upper bound.
        inclusive_max: If True, max_version is inclusive; otherwise exclusive.

    Returns:
        True if version is within the specified range.

    Example:
        >>> version_in_range("1.5.0", min_version="1.0.0", max_version="2.0.0")
        True
        >>> version_in_range("2.0.0", min_version="1.0.0", max_version="2.0.0")
        False
        >>> version_in_range("2.0.0", min_version="1.0.0", max_version="2.0.0", inclusive_max=True)
        True
    """
    try:
        v = Version(version)
    except ValueError:
        return False

    if min_version is not None:
        try:
            if v < Version(min_version):
                return False
        except ValueError:
            return False

    if max_version is not None:
        try:
            max_v = Version(max_version)
            if inclusive_max:
                if v > max_v:
                    return False
            else:
                if v >= max_v:
                    return False
        except ValueError:
            return False

    return True
