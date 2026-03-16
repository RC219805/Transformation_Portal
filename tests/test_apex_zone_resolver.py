"""Tests for APEX ZoneResolver.

Tests validate:
- Priority-based resolution
- Environment variable override
- Kubernetes detection
- AWS detection (mocked)
- Fallback behavior
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.metrics.zone_resolver import ZoneResolver


class TestZoneResolver:
    """Test ZoneResolver functionality."""

    def setup_method(self):
        """Clear zone cache before each test."""
        ZoneResolver.clear_cache()
        # Clear environment variables
        os.environ.pop("APEX_ZONE", None)
        os.environ.pop("KUBE_NODE_ZONE", None)

    def teardown_method(self):
        """Clean up after each test."""
        ZoneResolver.clear_cache()
        os.environ.pop("APEX_ZONE", None)
        os.environ.pop("KUBE_NODE_ZONE", None)

    def test_explicit_override(self):
        """Test explicit override has highest priority."""
        zone = ZoneResolver.resolve(override="test-zone-1")
        assert zone == "test-zone-1"

    def test_apex_zone_env_var(self):
        """Test APEX_ZONE environment variable."""
        os.environ["APEX_ZONE"] = "env-zone-1"
        zone = ZoneResolver.resolve()
        assert zone == "env-zone-1"

    def test_kube_node_zone_env_var(self):
        """Test Kubernetes KUBE_NODE_ZONE environment variable."""
        os.environ["KUBE_NODE_ZONE"] = "us-west-2a"
        zone = ZoneResolver.resolve()
        assert zone == "us-west-2a"

    def test_apex_zone_takes_priority_over_kube(self):
        """Test APEX_ZONE has priority over KUBE_NODE_ZONE."""
        os.environ["APEX_ZONE"] = "apex-zone"
        os.environ["KUBE_NODE_ZONE"] = "kube-zone"
        zone = ZoneResolver.resolve()
        assert zone == "apex-zone"

    def test_fallback_to_local(self):
        """Test fallback to 'local' when no zone detected."""
        zone = ZoneResolver.resolve()
        assert zone == "local"

    def test_caching(self):
        """Test zone resolution is cached."""
        os.environ["APEX_ZONE"] = "cached-zone"

        # First resolution
        zone1 = ZoneResolver.resolve()
        assert zone1 == "cached-zone"

        # Change environment variable (should still use cached value)
        os.environ["APEX_ZONE"] = "new-zone"
        zone2 = ZoneResolver.resolve()
        assert zone2 == "cached-zone"  # Still cached

        # Clear cache and resolve again
        ZoneResolver.clear_cache()
        zone3 = ZoneResolver.resolve()
        assert zone3 == "new-zone"  # Now uses new value

    def test_is_multi_zone(self):
        """Test is_multi_zone detection."""
        # Local zone is not multi-zone
        ZoneResolver.clear_cache()
        assert not ZoneResolver.is_multi_zone()

        # Non-local zone is multi-zone
        ZoneResolver.clear_cache()
        os.environ["APEX_ZONE"] = "us-west-2a"
        assert ZoneResolver.is_multi_zone()

    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    def test_kubernetes_downward_api(self, mock_open, mock_exists):
        """Test Kubernetes Downward API zone detection."""
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.read.return_value = "us-east-1b\n"
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file

        ZoneResolver.clear_cache()
        zone = ZoneResolver.resolve()

        assert zone == "us-east-1b"

    @patch("http.client.HTTPConnection")
    def test_aws_zone_detection(self, mock_http_connection):
        """Test AWS EC2 zone detection via IMDSv2."""
        # Mock responses for token and AZ requests
        mock_token_response = MagicMock()
        mock_token_response.status = 200
        mock_token_response.read.return_value = b"test-token-12345"

        mock_az_response = MagicMock()
        mock_az_response.status = 200
        mock_az_response.read.return_value = b"us-west-2c"

        # Mock connection instance
        mock_conn = MagicMock()
        mock_conn.getresponse.side_effect = [mock_token_response, mock_az_response]
        mock_http_connection.return_value = mock_conn

        ZoneResolver.clear_cache()
        zone = ZoneResolver.resolve()

        assert zone == "us-west-2c"

    @patch("http.client.HTTPConnection")
    def test_aws_detection_timeout(self, mock_http_connection):
        """Test AWS detection gracefully handles timeout."""
        # Simulate timeout
        mock_conn = MagicMock()
        mock_conn.request.side_effect = TimeoutError("timeout")
        mock_http_connection.return_value = mock_conn

        ZoneResolver.clear_cache()
        zone = ZoneResolver.resolve()

        # Should fallback to local
        assert zone == "local"
