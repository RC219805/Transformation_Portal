from __future__ import annotations

import pytest

from transformation_portal.depth.backends.registry import DepthBackendRegistry
from transformation_portal.lux_depth_v3._backend_contract import (
    backend_alias_warning,
    normalize_backend_id,
    normalize_backend_sequence,
)
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.manifest import BackendSelectionMetadata


def test_normalize_backend_id_warns_for_legacy_alias():
    with pytest.warns(FutureWarning, match="depth_anything_v3"):
        assert (
            normalize_backend_id(
                "depth_anything_v3",
                warn=True,
                warning_context="test",
            )
            == "da3"
        )


def test_normalize_backend_sequence_deduplicates_and_canonicalizes():
    assert normalize_backend_sequence(
        ("depth_anything_v3", "da3", "depth-anything-v3", "depth_pro"),
    ) == ("da3", "depth_pro")


def test_enhance_config_normalizes_legacy_backend_alias_and_chain():
    with pytest.warns(FutureWarning, match="depth_anything_v3"):
        config = EnhanceConfig(
            depth_backend="depth_anything_v3",
            depth_operational_fallback_chain=(
                "depth-anything-v3",
                "da2",
                "da3",
            ),
        )

    assert config.depth_backend == "da3"
    assert config.depth_operational_fallback_chain == ("da3", "da2")


def test_backend_selection_metadata_normalizes_legacy_aliases():
    metadata = BackendSelectionMetadata(
        requested_backend="depth_anything_v3",
        resolved_backend="depth-anything-v3",
        resolution_status="success",
        resolution_reason=backend_alias_warning("depth_anything_v3", "da3"),
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        device="cpu",
        attempts=[{"backend": "depth_anything_v3", "status": "success"}],
    )

    assert metadata.requested_backend == "da3"
    assert metadata.resolved_backend == "da3"
    assert metadata.attempts == [{"backend": "da3", "status": "success"}]


def test_registry_accepts_legacy_backend_aliases(monkeypatch):
    registry = DepthBackendRegistry()
    sentinel_backend = object()
    monkeypatch.setitem(DepthBackendRegistry._backends, "da3", sentinel_backend)

    assert registry.has_backend("depth_anything_v3") is True
    assert registry.get_backend_class("depth-anything-v3") is sentinel_backend
