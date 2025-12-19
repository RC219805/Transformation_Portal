"""Tests for DA3 model versioning and license validation.

Tests Sprint 1 integration: v1.1 model support and license validation.
"""

import pytest
import warnings
from urllib.parse import urlparse

from lux_depth_v3.config import ModelVariant, ModelInfo, ModelLicense
from lux_depth_v3.license import (
    LicenseValidator,
    LicenseViolationWarning,
    validate_license,
)


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_basic(self):
        """Test basic ModelInfo attributes."""
        info = ModelInfo(
            name="DA3-TEST",
            params="0.5B",
            license=ModelLicense.APACHE_2_0,
            huggingface_id="depth-anything/DA3-TEST",
        )

        assert info.name == "DA3-TEST"
        assert info.params == "0.5B"
        assert info.license == ModelLicense.APACHE_2_0
        assert info.huggingface_id == "depth-anything/DA3-TEST"
        assert info.version is None
        assert info.is_commercial

    def test_model_info_with_version(self):
        """Test ModelInfo with version."""
        info = ModelInfo(
            name="DA3-TEST",
            params="0.5B",
            license=ModelLicense.CC_BY_NC_4_0,
            huggingface_id="depth-anything/DA3-TEST-1.1",
            version="1.1",
        )

        assert info.version == "1.1"
        assert info.display_name == "DA3-TEST-1.1"
        assert not info.is_commercial

    def test_model_info_display_name_no_version(self):
        """Test display name without version."""
        info = ModelInfo(
            name="DA3-TEST",
            params="0.5B",
            license=ModelLicense.APACHE_2_0,
            huggingface_id="depth-anything/DA3-TEST",
        )

        assert info.display_name == "DA3-TEST"

    def test_model_info_commercial_apache(self):
        """Test commercial check for Apache license."""
        info = ModelInfo(
            name="DA3-TEST",
            params="0.5B",
            license=ModelLicense.APACHE_2_0,
            huggingface_id="depth-anything/DA3-TEST",
        )

        assert info.is_commercial

    def test_model_info_commercial_cc_nc(self):
        """Test commercial check for CC-BY-NC license."""
        info = ModelInfo(
            name="DA3-TEST",
            params="0.5B",
            license=ModelLicense.CC_BY_NC_4_0,
            huggingface_id="depth-anything/DA3-TEST",
        )

        assert not info.is_commercial


class TestModelVariant:
    """Test ModelVariant enum."""

    def test_v1_1_variants_exist(self):
        """Test that all v1.1 variants are defined."""
        assert hasattr(ModelVariant, "DA3_NESTED_GIANT_LARGE_V1_1")
        assert hasattr(ModelVariant, "DA3_GIANT_V1_1")
        assert hasattr(ModelVariant, "DA3_LARGE_V1_1")

    def test_v1_0_variants_exist(self):
        """Test that deprecated v1.0 variants still exist."""
        assert hasattr(ModelVariant, "DA3_NESTED_GIANT_LARGE")
        assert hasattr(ModelVariant, "DA3_GIANT")
        assert hasattr(ModelVariant, "DA3_LARGE")

    def test_apache_variants_exist(self):
        """Test that Apache-licensed variants exist."""
        assert hasattr(ModelVariant, "DA3_BASE")
        assert hasattr(ModelVariant, "DA3_SMALL")
        assert hasattr(ModelVariant, "DA3_METRIC_LARGE")
        assert hasattr(ModelVariant, "DA3_MONO_LARGE")

    def test_variant_info_property(self):
        """Test that info property returns ModelInfo."""
        variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
        info = variant.info

        assert isinstance(info, ModelInfo)
        assert info.name == "DA3NESTED-GIANT-LARGE"
        assert info.version == "1.1"
        assert info.license == ModelLicense.CC_BY_NC_4_0

    def test_get_recommended(self):
        """Test get_recommended() returns v1.1 nested model."""
        recommended = ModelVariant.get_recommended()

        assert recommended == ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
        assert recommended.info.version == "1.1"

    def test_commercial_alternative_nc_to_apache(self):
        """Test commercial alternative mapping for NC models."""
        nc_variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
        commercial = ModelVariant.get_commercial_alternative(nc_variant)

        assert commercial == ModelVariant.DA3_METRIC_LARGE
        assert commercial.info.is_commercial

    def test_commercial_alternative_apache_returns_self(self):
        """Test commercial alternative for Apache model returns itself."""
        apache_variant = ModelVariant.DA3_BASE
        commercial = ModelVariant.get_commercial_alternative(apache_variant)

        assert commercial == apache_variant

    def test_commercial_alternative_giant_v1_1(self):
        """Test commercial alternative for GIANT v1.1."""
        variant = ModelVariant.DA3_GIANT_V1_1
        commercial = ModelVariant.get_commercial_alternative(variant)

        assert commercial == ModelVariant.DA3_BASE
        assert commercial.info.is_commercial

    def test_commercial_alternative_large_v1_1(self):
        """Test commercial alternative for LARGE v1.1."""
        variant = ModelVariant.DA3_LARGE_V1_1
        commercial = ModelVariant.get_commercial_alternative(variant)

        assert commercial == ModelVariant.DA3_BASE
        assert commercial.info.is_commercial


class TestLicenseValidator:
    """Test LicenseValidator class."""

    def test_check_commercial_use_non_commercial_allowed(self):
        """Test non-commercial use is always allowed."""
        validator = LicenseValidator()
        variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1

        # Non-commercial use should be allowed even for NC-licensed models
        assert validator.check_commercial_use(variant, commercial_use=False, warn=False)

    def test_check_commercial_use_apache_allowed(self):
        """Test commercial use allowed for Apache models."""
        validator = LicenseValidator()
        variant = ModelVariant.DA3_BASE

        assert validator.check_commercial_use(variant, commercial_use=True, warn=False)

    def test_check_commercial_use_nc_denied(self):
        """Test commercial use denied for NC-licensed models."""
        validator = LicenseValidator()
        variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1

        # Should return False for commercial use of NC model
        assert not validator.check_commercial_use(variant, commercial_use=True, warn=False)

    def test_check_commercial_use_warning(self):
        """Test warning is issued for commercial use of NC model."""
        validator = LicenseValidator()
        variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1

        with pytest.warns(LicenseViolationWarning):
            validator.check_commercial_use(variant, commercial_use=True, warn=True)

    def test_get_license_info_apache(self):
        """Test get_license_info for Apache model."""
        validator = LicenseValidator()
        variant = ModelVariant.DA3_BASE
        info = validator.get_license_info(variant)

        assert info["model"] == "DA3-BASE"
        assert info["license"] == "Apache-2.0"
        assert info["commercial_allowed"]
        # Fix: Use urlparse for robust hostname check
        parsed_url = urlparse(info["license_url"])
        assert parsed_url.hostname in ("apache.org", "www.apache.org")
        assert info["alternative"] is None  # Already commercial

    def test_get_license_info_nc(self):
        """Test get_license_info for NC model."""
        validator = LicenseValidator()
        variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
        info = validator.get_license_info(variant)

        assert info["model"] == "DA3NESTED-GIANT-LARGE-1.1"
        assert info["license"] == "CC-BY-NC-4.0"
        assert not info["commercial_allowed"]
        # Fix: Use urlparse for robust hostname check
        parsed_url = urlparse(info["license_url"])
        assert parsed_url.hostname in ("creativecommons.org", "www.creativecommons.org")
        assert info["alternative"] == "DA3METRIC-LARGE"

    def test_get_license_info_capabilities(self):
        """Test license info includes capabilities."""
        validator = LicenseValidator()
        variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
        info = validator.get_license_info(variant)

        assert "capabilities" in info
        assert info["capabilities"]["relative_depth"]
        assert info["capabilities"]["metric_depth"]
        assert info["capabilities"]["gaussian_splatting"]


class TestValidateLicense:
    """Test validate_license function."""

    def test_validate_license_non_commercial_no_warning(self):
        """Test validate_license for non-commercial use."""
        variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1

        # Should not raise or warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_license(variant, commercial_use=False, strict=False)

    def test_validate_license_apache_commercial_allowed(self):
        """Test validate_license for commercial use of Apache model."""
        variant = ModelVariant.DA3_BASE

        # Should not raise or warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_license(variant, commercial_use=True, strict=False)

    def test_validate_license_nc_commercial_warning(self):
        """Test validate_license warns for commercial use of NC model."""
        variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1

        with pytest.warns(LicenseViolationWarning):
            validate_license(variant, commercial_use=True, strict=False)

    def test_validate_license_nc_commercial_strict_raises(self):
        """Test validate_license strict mode raises for NC model."""
        variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1

        with pytest.raises(RuntimeError, match="cannot be used for commercial"):
            validate_license(variant, commercial_use=True, strict=True)

    def test_validate_license_strict_error_message(self):
        """Test strict error message includes alternative."""
        variant = ModelVariant.DA3_LARGE_V1_1

        with pytest.raises(RuntimeError) as exc_info:
            validate_license(variant, commercial_use=True, strict=True)

        error_msg = str(exc_info.value)
        assert "DA3-LARGE-1.1" in error_msg
        assert "DA3-BASE" in error_msg or "DA3METRIC-LARGE" in error_msg


class TestBackwardCompatibility:
    """Test backward compatibility with legacy model names."""

    def test_legacy_variants_exist(self):
        """Test legacy string-based variants still exist."""
        assert hasattr(ModelVariant, "NESTED_GIANT_LARGE")
        assert hasattr(ModelVariant, "GIANT")
        assert hasattr(ModelVariant, "LARGE")
        assert hasattr(ModelVariant, "BASE")
        assert hasattr(ModelVariant, "SMALL")
        assert hasattr(ModelVariant, "METRIC_LARGE")
        assert hasattr(ModelVariant, "MONO_LARGE")

    def test_legacy_variant_info(self):
        """Test legacy variants have correct info."""
        variant = ModelVariant.NESTED_GIANT_LARGE
        info = variant.info

        assert info.name == "DA3NESTED-GIANT-LARGE"
        assert info.version == "1.0"  # Legacy is v1.0
        assert info.license == ModelLicense.CC_BY_NC_4_0

    def test_commercial_alternative_legacy(self):
        """Test commercial alternatives work for legacy variants."""
        variant = ModelVariant.NESTED_GIANT_LARGE
        commercial = ModelVariant.get_commercial_alternative(variant)

        assert commercial == ModelVariant.DA3_METRIC_LARGE
        assert commercial.info.is_commercial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
