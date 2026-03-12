"""Tests for backend selection metadata (ADR-023 Phase 3)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.transformation_portal.lux_depth_v3.manifest import BackendSelectionMetadata, CombinedManifest


def test_backend_selection_metadata_schema():
    """Test BackendSelectionMetadata schema."""
    metadata = BackendSelectionMetadata(
        requested_backend="da3",
        resolved_backend="da3",
        resolution_status="success",
        resolution_reason=None,
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        device="cpu",
    )

    assert metadata.requested_backend == "da3"
    assert metadata.resolved_backend == "da3"
    assert metadata.resolution_status == "success"
    assert metadata.resolution_reason is None
    assert metadata.schema_version == "1.0"


def test_backend_selection_serialization_roundtrip():
    """Test BackendSelectionMetadata serialization/deserialization."""
    metadata = BackendSelectionMetadata(
        requested_backend="depth_pro",
        resolved_backend="da3",
        resolution_status="fallback",
        resolution_reason="Requested 'depth_pro' not available",
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        device="cpu",
    )

    # Serialize
    data = metadata.to_dict()
    assert isinstance(data, dict)
    assert data["requested_backend"] == "depth_pro"
    assert data["resolution_status"] == "fallback"

    # Deserialize
    loaded = BackendSelectionMetadata.from_dict(data)
    assert loaded.requested_backend == metadata.requested_backend
    assert loaded.resolved_backend == metadata.resolved_backend
    assert loaded.resolution_status == metadata.resolution_status
    assert loaded.resolution_reason == metadata.resolution_reason


def test_backend_selection_metadata_success_path():
    """Test backend selection when requested matches resolved."""
    metadata = BackendSelectionMetadata(
        requested_backend=None,  # Auto-select
        resolved_backend="da3",
        resolution_status="success",
        resolution_reason=None,
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        device="cpu",
    )

    assert metadata.resolution_status == "success"
    assert metadata.resolution_reason is None


def test_backend_selection_metadata_explicit_da3():
    """Test backend selection when user explicitly requests DA3."""
    metadata = BackendSelectionMetadata(
        requested_backend="da3",
        resolved_backend="da3",
        resolution_status="success",
        resolution_reason=None,
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        device="mps",
    )

    assert metadata.requested_backend == "da3"
    assert metadata.resolved_backend == "da3"
    assert metadata.resolution_status == "success"


def test_backend_selection_metadata_fallback():
    """Test backend selection when fallback occurs."""
    metadata = BackendSelectionMetadata(
        requested_backend="depth_pro",
        resolved_backend="da3",
        resolution_status="fallback",
        resolution_reason="Requested 'depth_pro' not available, using 'da3'",
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        device="cpu",
    )

    assert metadata.requested_backend == "depth_pro"
    assert metadata.resolved_backend == "da3"
    assert metadata.resolution_status == "fallback"
    assert "using 'da3'" in metadata.resolution_reason


def test_manifest_includes_backend_selection(tmp_path):
    """Test CombinedManifest includes backend_selection field."""
    backend_metadata = BackendSelectionMetadata(
        requested_backend="da3",
        resolved_backend="da3",
        resolution_status="success",
        resolution_reason=None,
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        device="cpu",
    )

    manifest = CombinedManifest(backend_selection=backend_metadata)

    # Serialize to dict
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)

    # Read back and verify
    with open(manifest_path) as f:
        data = json.load(f)

    assert "backend_selection" in data
    assert data["backend_selection"]["resolved_backend"] == "da3"
    assert data["backend_selection"]["resolution_status"] == "success"


def test_manifest_backward_compatible(tmp_path):
    """Test CombinedManifest handles missing backend_selection (backward compatibility)."""
    # Create old manifest without backend_selection
    manifest_path = tmp_path / "old_manifest.json"
    old_data = {
        "input": {
            "image_path": "test.jpg",
            "schema_version": "1.0",
        },
        "timing": {
            "depth_seconds": 10.0,
            "v2_seconds": 5.0,
            "total_seconds": 15.0,
            "timestamp_utc": "2026-01-01T00:00:00Z",
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(old_data, f)

    # Load old manifest
    manifest = CombinedManifest.load(manifest_path)

    assert manifest.backend_selection is None  # Should be None for old manifests
    assert manifest.input is not None
    assert manifest.timing is not None


def test_backend_selection_unsupported_schema():
    """Test that unsupported schema version raises ValueError."""
    data = {
        "requested_backend": "da3",
        "resolved_backend": "da3",
        "resolution_status": "success",
        "resolution_reason": None,
        "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "device": "cpu",
        "schema_version": "2.0",  # Unsupported version
    }

    with pytest.raises(ValueError, match="Unsupported BackendSelectionMetadata schema"):
        BackendSelectionMetadata.from_dict(data)


def test_manifest_serialization_with_backend_selection(tmp_path):
    """Test full manifest serialization roundtrip with backend_selection."""
    backend_metadata = BackendSelectionMetadata(
        requested_backend="da3",
        resolved_backend="da3",
        resolution_status="success",
        resolution_reason=None,
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        device="cpu",
    )

    manifest = CombinedManifest(backend_selection=backend_metadata)

    # Save and load
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    loaded = CombinedManifest.load(manifest_path)

    assert loaded.backend_selection is not None
    assert loaded.backend_selection.resolved_backend == "da3"
    assert loaded.backend_selection.resolution_status == "success"


def test_backend_selection_from_legacy_manifest_alias_normalizes_to_canonical():
    """Legacy alias values should deserialize to canonical backend IDs."""
    metadata = BackendSelectionMetadata.from_dict(
        {
            "requested_backend": "depth_anything_v3",
            "resolved_backend": "depth-anything-v3",
            "resolution_status": "success",
            "resolution_reason": None,
            "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "device": "cpu",
            "schema_version": "1.0",
        }
    )

    assert metadata.requested_backend == "da3"
    assert metadata.resolved_backend == "da3"
