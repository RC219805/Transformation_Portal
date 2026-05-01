"""Tests for metrics.zone_resolver — ZoneResolver class methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.metrics.zone_resolver import ZoneResolver

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_zone_cache():
    """Ensure each test starts with a clean cache."""
    ZoneResolver.clear_cache()
    yield
    ZoneResolver.clear_cache()


class TestZoneResolverResolve:
    def test_override_returns_override_value(self):
        """Explicit override takes highest priority."""
        assert ZoneResolver.resolve(override="test-zone") == "test-zone"

    def test_override_skips_cache(self):
        """Override always returns the given value, ignoring cache."""
        ZoneResolver.resolve()  # populate cache
        assert ZoneResolver.resolve(override="explicit-zone") == "explicit-zone"

    def test_apex_zone_env_var_used(self, monkeypatch):
        """APEX_ZONE environment variable is the second priority."""
        monkeypatch.setenv("APEX_ZONE", "eu-west-1")
        monkeypatch.delenv("KUBE_NODE_ZONE", raising=False)
        result = ZoneResolver.resolve()
        assert result == "eu-west-1"

    def test_kube_node_zone_env_var_used(self, monkeypatch):
        """KUBE_NODE_ZONE is used when APEX_ZONE is absent."""
        monkeypatch.delenv("APEX_ZONE", raising=False)
        monkeypatch.setenv("KUBE_NODE_ZONE", "zone-k8s-1")
        result = ZoneResolver.resolve()
        assert result == "zone-k8s-1"

    def test_fallback_to_local_when_no_env_and_no_network(self, monkeypatch):
        """Falls back to 'local' when no env vars and AWS/k8s not reachable."""
        monkeypatch.delenv("APEX_ZONE", raising=False)
        monkeypatch.delenv("KUBE_NODE_ZONE", raising=False)

        # Mock AWS metadata to be unreachable
        with patch(
            "transformation_portal.metrics.zone_resolver.ZoneResolver._resolve_aws_zone",
            return_value=None,
        ):
            result = ZoneResolver.resolve()

        assert result == "local"

    def test_cache_prevents_re_resolution(self, monkeypatch):
        """resolve() called twice uses cached value on second call."""
        monkeypatch.setenv("APEX_ZONE", "cached-zone")
        monkeypatch.delenv("KUBE_NODE_ZONE", raising=False)

        first = ZoneResolver.resolve()
        # Remove env var — cache should still return first result
        monkeypatch.delenv("APEX_ZONE", raising=False)
        second = ZoneResolver.resolve()

        assert first == second == "cached-zone"


class TestZoneResolverIsMultiZone:
    def test_local_is_not_multi_zone(self, monkeypatch):
        """Zone 'local' → is_multi_zone() returns False."""
        monkeypatch.delenv("APEX_ZONE", raising=False)
        monkeypatch.delenv("KUBE_NODE_ZONE", raising=False)
        with patch(
            "transformation_portal.metrics.zone_resolver.ZoneResolver._resolve_aws_zone",
            return_value=None,
        ):
            assert ZoneResolver.is_multi_zone() is False

    def test_non_local_zone_is_multi_zone(self, monkeypatch):
        """Zone 'us-west-2a' → is_multi_zone() returns True."""
        monkeypatch.setenv("APEX_ZONE", "us-west-2a")
        assert ZoneResolver.is_multi_zone() is True


class TestZoneResolverAWS:
    def test_aws_network_failure_returns_none(self):
        """Network OSError during AWS metadata → _resolve_aws_zone returns None."""
        import http.client

        with patch.object(http.client.HTTPConnection, "__init__", side_effect=OSError("refused")):
            result = ZoneResolver._resolve_aws_zone()
        assert result is None

    def test_aws_bad_status_returns_none(self):
        """Non-2xx response from AWS metadata → _resolve_aws_zone returns None."""
        mock_resp = MagicMock()
        mock_resp.status = 404
        mock_resp.read.return_value = b""
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with patch("http.client.HTTPConnection", return_value=mock_conn):
            result = ZoneResolver._resolve_aws_zone()
        assert result is None


class TestClearCache:
    def test_clear_resets_cached_zone(self, monkeypatch):
        """clear_cache() forces a fresh resolution on next call."""
        monkeypatch.setenv("APEX_ZONE", "zone-a")
        monkeypatch.delenv("KUBE_NODE_ZONE", raising=False)
        ZoneResolver.resolve()  # sets cache to "zone-a"

        monkeypatch.setenv("APEX_ZONE", "zone-b")
        ZoneResolver.clear_cache()

        result = ZoneResolver.resolve()
        assert result == "zone-b"
