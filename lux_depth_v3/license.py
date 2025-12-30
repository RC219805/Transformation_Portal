"""License validation and warnings for DA3 models.

This module provides comprehensive license compliance checking for
Depth Anything 3 model variants, ensuring users are aware of
licensing restrictions and commercial use limitations.
"""

from __future__ import annotations

import warnings
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lux_depth_v3.config import ModelVariant

from lux_depth_v3.config import ModelLicense


class LicenseViolationWarning(UserWarning):
    """Warning for potential license violations."""

    pass


class LicenseValidator:
    """Validates model license compliance.

    Provides methods to check commercial use permissions and generate
    appropriate warnings or errors when license restrictions are violated.
    """

    @staticmethod
    def check_commercial_use(variant: "ModelVariant", commercial_use: bool = False, warn: bool = True) -> bool:
        """Check if model can be used commercially.

        Args:
            variant: Model variant to check
            commercial_use: Whether this is commercial use
            warn: Whether to issue warnings

        Returns:
            True if usage is allowed, False otherwise
        """
        from lux_depth_v3.config import ModelVariant

        info = variant.info

        if not commercial_use:
            # Non-commercial use is always OK
            return True

        if info.license == ModelLicense.CC_BY_NC_4_0:
            if warn:
                alternative = ModelVariant.get_commercial_alternative(variant)
                alt_name = alternative.info.display_name if alternative else "DA3METRIC-LARGE"
                alt_license = alternative.info.license.value if alternative else "Apache-2.0"

                msg = (
                    f"\n{'=' * 70}\n"
                    f"⚠️  LICENSE WARNING: {info.display_name}\n"
                    f"{'=' * 70}\n"
                    f"License: {info.license.value} (Non-Commercial)\n"
                    f"Commercial use is NOT permitted.\n\n"
                    f"For commercial applications, use:\n"
                    f"  → {alt_name} ({alt_license})\n\n"
                    f"More info: https://creativecommons.org/licenses/by-nc/4.0/\n"
                    f"{'=' * 70}\n"
                )
                warnings.warn(msg, LicenseViolationWarning, stacklevel=3)
            return False

        return True

    def get_commercial_alternative(self, variant: "ModelVariant") -> Optional["ModelVariant"]:
        """Get commercial-friendly alternative for NC-licensed models.

        Args:
            variant: Model variant to check

        Returns:
            Commercial alternative variant, or None if already commercial
        """
        from lux_depth_v3.config import ModelVariant

        return ModelVariant.get_commercial_alternative(variant)

    def get_license_info(self, variant: "ModelVariant") -> dict:
        """Get detailed license information.

        Args:
            variant: Model variant to query

        Returns:
            Dictionary with license details, URLs, and alternatives
        """
        from lux_depth_v3.config import ModelVariant

        info = variant.info
        alternative = ModelVariant.get_commercial_alternative(variant)

        return {
            "model": info.display_name,
            "license": info.license.value,
            "commercial_allowed": info.is_commercial,
            "license_url": (
                "https://www.apache.org/licenses/LICENSE-2.0"
                if info.license == ModelLicense.APACHE_2_0
                else "https://creativecommons.org/licenses/by-nc/4.0/"
            ),
            "alternative": (alternative.info.display_name if alternative and not info.is_commercial else None),
            "capabilities": info.capabilities or {},
        }


def validate_license(variant: "ModelVariant", commercial_use: bool = False, strict: bool = False) -> None:
    """Validate license and issue warnings.

    Args:
        variant: Model variant to check
        commercial_use: Whether this is commercial use
        strict: If True, raise exception instead of warning

    Raises:
        RuntimeError: If strict=True and license is violated
    """
    from lux_depth_v3.config import ModelVariant

    validator = LicenseValidator()

    if not validator.check_commercial_use(variant, commercial_use, warn=not strict):
        if strict:
            alternative = ModelVariant.get_commercial_alternative(variant)
            alt_name = alternative.info.display_name if alternative else "DA3METRIC-LARGE"

            raise RuntimeError(
                f"Model {variant.info.display_name} ({variant.info.license.value}) "
                f"cannot be used for commercial purposes. "
                f"Use {alt_name} instead."
            )


def get_license_info(variant: "ModelVariant") -> dict:
    """Get detailed license information.

    Args:
        variant: Model variant to query

    Returns:
        Dictionary with license details, URLs, and alternatives
    """
    validator = LicenseValidator()
    return validator.get_license_info(variant)


def get_commercial_alternative(variant: "ModelVariant") -> Optional["ModelVariant"]:
    """Get commercial-friendly alternative for NC-licensed models.

    Args:
        variant: Model variant to check

    Returns:
        Commercial alternative variant, or None if already commercial
    """
    from lux_depth_v3.config import ModelVariant

    return ModelVariant.get_commercial_alternative(variant)
