"""Version checking and compatibility utilities."""

import re
from typing import Any, Tuple, Union


class Version:
    """Simple semantic version parser and comparator."""

    def __init__(self, version_str: str) -> None:
        self.raw = version_str
        self.major, self.minor, self.patch, self.prerelease = self._parse(version_str)

    def _parse(self, v: str) -> Tuple[int, int, int, str]:
        """Parse version string (e.g., '1.2.3', '2.0.0-beta')."""
        # Regex for SemVer-ish strings
        match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:[.-](.+))?$", v)
        if not match:
            raise ValueError(f"Invalid version string: {v}")
        
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3) or 0)
        prerelease = match.group(4) or ""
        
        return major, minor, patch, prerelease

    def __repr__(self) -> str:
        return f"Version('{self.raw}')"

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __ge__(self, other: Any) -> bool:
        return not self < other


def check_version_compatibility(
    current_version: str, required_version: str
) -> bool:
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


def require_version(min_version: str) -> None:
    """Raise RuntimeError if package version is too old.
    
    Useful for ensuring plugins define a minimum required version of 
    Transformation Portal.
    """
    # Import here to avoid circular dependency
    from transformation_portal import __version__ as current_ver
    
    if not check_version_compatibility(current_ver, min_version):
        raise RuntimeError(
            f"Transformation Portal v{min_version}+ required. "
            f"Found v{current_ver}."
        )
