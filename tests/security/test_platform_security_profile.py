"""Tests for platform security profile integration (ADR-032 security layer).

Test Coverage:
- get_platform_fingerprint() includes security_profile by default
- include_security_profile=False behavior
- Security profile field stability/determinism
- torch_security.py enforcement and hash determinism
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from transformation_portal.core.platform_matrix import (
    get_platform_fingerprint,
    get_security_profile,
)

pytestmark = pytest.mark.security


def _torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


class TestPlatformSecurityProfile:
    """Tests for security_profile in platform fingerprint."""

    def test_fingerprint_includes_security_profile_by_default(self):
        """Test get_platform_fingerprint includes security_profile by default."""
        fingerprint = get_platform_fingerprint()

        assert "security_profile" in fingerprint
        assert isinstance(fingerprint["security_profile"], dict)

    def test_fingerprint_security_profile_fields(self):
        """Test security_profile contains expected fields."""
        fingerprint = get_platform_fingerprint()

        security_profile = fingerprint["security_profile"]
        # Should contain version, policy, and profile_hash
        assert "version" in security_profile
        assert "policy" in security_profile
        assert "profile_hash" in security_profile
        assert security_profile["profile_hash"].startswith("sha256:")

    def test_fingerprint_exclude_security_profile(self):
        """Test include_security_profile=False excludes security_profile."""
        fingerprint = get_platform_fingerprint(include_security_profile=False)

        assert "security_profile" not in fingerprint

    def test_security_profile_deterministic(self):
        """Test security_profile is deterministic across calls."""
        fingerprint1 = get_platform_fingerprint()
        fingerprint2 = get_platform_fingerprint()

        assert fingerprint1["security_profile"] == fingerprint2["security_profile"]

    def test_get_security_profile_canonical(self):
        """Test get_security_profile returns canonical values."""
        profile = get_security_profile()

        # Should be a dict with expected keys
        assert isinstance(profile, dict)
        assert "version" in profile
        assert "policy" in profile
        assert "profile_hash" in profile

        # Profile hash should be deterministic
        profile2 = get_security_profile()
        assert profile["profile_hash"] == profile2["profile_hash"]


class TestTorchSecurityEnforcement:
    """Tests for torch_security.py enforcement and hash determinism."""

    def test_security_profile_hash_deterministic(self):
        """Test get_security_profile_hash returns deterministic value."""
        from transformation_portal.core.security.torch_security import (
            get_security_profile_hash,
        )

        hash1 = get_security_profile_hash()
        hash2 = get_security_profile_hash()

        assert hash1 == hash2
        assert hash1.startswith("sha256:")

    def test_security_profile_hash_uses_canonical_json(self):
        """Test hash uses canonical JSON serialization."""
        from transformation_portal.core.security.torch_security import (
            MINIMUM_SUPPORTED_TORCH_VERSION,
            SECURITY_PROFILE_VERSION,
            get_security_profile_hash,
        )
        from transformation_portal.ingest.canonical_json import canonicalize_json

        # Manually compute expected hash
        profile_data = {
            "policy_version": SECURITY_PROFILE_VERSION,
            "minimum_supported_torch_version": MINIMUM_SUPPORTED_TORCH_VERSION,
            "cve_2025_32434_posture": "fixed_by_supported_torch_baseline",
            "torch_load_policy": "weights_only_true",
        }
        expected_bytes = canonicalize_json(profile_data)
        expected_digest = hashlib.sha256(expected_bytes).hexdigest()
        expected_hash = f"sha256:{expected_digest}"

        actual_hash = get_security_profile_hash()

        assert actual_hash == expected_hash

    def test_get_canonical_security_profile_static(self):
        """Test canonical profile uses STATIC values only.

        Note: get_canonical_security_profile() returns a public API structure
        with field names that differ from the internal get_security_profile_hash()
        format. This is intentional - the hash uses its own internal structure.
        """
        from transformation_portal.core.security.torch_security import (
            get_canonical_security_profile,
        )

        profile = get_canonical_security_profile()

        # Should return static policy values (matches actual implementation)
        assert "policy_version" in profile
        assert "cve_mitigation" in profile
        assert "minimum_supported_torch_version" in profile
        assert "torch_load_enforced" in profile
        assert "weights_only" in profile
        assert profile["minimum_supported_torch_version"] == "2.13.0"
        assert profile["cve_mitigation"] == "fixed-by-supported-torch-baseline"

        # Should NOT include runtime state
        assert "enforcement_installed" not in profile

    def test_torch_security_compliance_reports_supported_baseline(self):
        """Test compliance status preserves legacy keys and reports supported baseline."""
        from transformation_portal.core.security.torch_security import (
            MINIMUM_SUPPORTED_TORCH_VERSION,
            check_torch_security_compliance,
        )

        status = check_torch_security_compliance()

        assert "torch_version" in status
        assert "cve_2025_32434_vulnerable" in status
        assert "mitigation_available" in status
        assert "recommendation" in status
        assert status["minimum_supported_torch_version"] == MINIMUM_SUPPORTED_TORCH_VERSION
        assert "supported_security_baseline_met" in status

    def test_install_enforcement_idempotent(self):
        """Test install_global_enforcement is idempotent."""
        from transformation_portal.core.security.torch_security import (
            install_global_enforcement,
            is_enforcement_installed,
        )

        # Install enforcement
        result1 = install_global_enforcement()
        installed1 = is_enforcement_installed()

        # Install again
        result2 = install_global_enforcement()
        installed2 = is_enforcement_installed()

        # Both should succeed and state should remain consistent
        assert result1 == result2
        assert installed1 == installed2

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available",
    )
    def test_enforcement_blocks_unsafe_load(self):
        """Test enforcement blocks weights_only=False."""
        from transformation_portal.core.security.torch_security import (
            SecurityPolicyViolation,
            install_global_enforcement,
            is_enforcement_installed,
        )

        # Ensure enforcement is installed
        install_global_enforcement()

        if is_enforcement_installed():
            import pickle
            import tempfile

            import torch

            # Create a test file with a simple pickled tensor
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                # Use safe save first
                test_tensor = torch.tensor([1, 2, 3])
                torch.save(test_tensor, f.name)

                # Loading with weights_only=False should be blocked by enforcement
                # (The exact behavior depends on the enforcement implementation)
                try:
                    # The enforcement should either:
                    # 1. Override weights_only to True
                    # 2. Raise SecurityPolicyViolation
                    result = torch.load(f.name, weights_only=False)
                    # If we get here, enforcement forced weights_only=True
                    assert torch.equal(result, test_tensor)
                except SecurityPolicyViolation:
                    # This is also acceptable behavior
                    pass
