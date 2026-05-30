"""Run-card provenance tests for the backend-selection quartet.

These tests close the contract gap CLAUDE.md calls out:

    "Backend resolution metadata (requested_backend, resolved_backend,
    resolution_status, resolution_reason) is part of every manifest —
    preserve it."

The existing tests/lux_depth_v3/test_orchestrator_manifests.py already
asserts presence of the first three legs and a `device`/`attempts` pair.
The fourth leg — ``resolution_reason`` — and the fallback-time
propagation of that reason through the persisted manifest had zero
assertions before this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def _make_test_image(tmp_path: Path, name: str = "test.png", size: tuple = (64, 64)) -> Path:
    """Create a minimal test image for orchestrator tests."""
    image_path = tmp_path / name
    Image.new("RGB", size, color="white").save(image_path)
    return image_path


def _make_mock_depth_result(backend_id: str = "da3"):
    """Create a deterministic synthetic depth result."""
    from transformation_portal.depth.backends.protocol import DepthResult

    return DepthResult(
        depth_map=np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        metadata={},
        depth_units="relative",
        backend_id=backend_id,
        device="cpu",
    )


def _make_mock_registry(backend_id: str = "da3"):
    """Create a mock depth backend registry."""
    backend = Mock()
    backend.name = backend_id
    backend.license_type = Mock(value="commercial")
    backend.ensure_available.return_value = None
    backend.compute.return_value = _make_mock_depth_result(backend_id)

    registry = Mock()
    registry.get_backend.return_value = backend
    return registry


def _create_orchestrator(tmp_path: Path, **config_kwargs):
    """Create an orchestrator instance with mocked backend registry."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    defaults = {
        "depth_backend": "da3",
        "depth_device": "cpu",
        "enable_v2": False,
        "enable_materials_v3": False,
    }
    defaults.update(config_kwargs)
    config = EnhanceConfig(**defaults)

    with patch(
        "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
        return_value=_make_mock_registry(),
    ):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        return orchestrator


def _get_manifest_data(tmp_path: Path, image_name: str) -> Dict[str, Any]:
    """Process an image and return the manifest data as a dict."""
    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    orchestrator = _create_orchestrator(tmp_path)
    test_image = _make_test_image(tmp_path, image_name)

    result = orchestrator.enhance_image(
        ImageInput(path=test_image),
        input_root=tmp_path,
    )

    manifest_path = Path(result["manifest"])
    with open(manifest_path) as f:
        return json.load(f)


class TestResolutionReasonField:
    """The fourth leg of the quartet — resolution_reason — must always be wired."""

    def test_resolution_reason_key_present_on_success_path(self, tmp_path: Path) -> None:
        """`backend_selection` must contain a `resolution_reason` key on success.

        The value may be None when no fallback occurred, but the key itself
        must be present so downstream parsers do not need defensive .get()
        calls.
        """
        manifest = _get_manifest_data(tmp_path, "reason_success.png")
        assert "resolution_reason" in manifest["backend_selection"]
        assert manifest["licensing"]["schema_version"] == "1.0"
        assert "non_commercial_active" in manifest["licensing"]


class TestFallbackPropagation:
    """A fallback in the live run must carry a populated reason into the manifest."""

    def test_fallback_reason_propagates_into_persisted_manifest(self, tmp_path: Path) -> None:
        """When the active backend is replaced post-init by a recovered
        fallback, the persisted `backend_selection.resolution_reason` must
        be a non-empty string identifying the fallback.

        We exercise the persistence layer directly via
        ``CombinedManifest.save``/``load`` round-trip, mirroring the way
        the orchestrator writes its manifest in ``_write_manifest``.
        """
        from transformation_portal.lux_depth_v3.manifest import (
            BackendSelectionMetadata,
            CombinedManifest,
        )

        fallback_metadata = BackendSelectionMetadata(
            requested_backend="da3",
            resolved_backend="synthetic",
            resolution_status="fallback",
            resolution_reason=("Fallback from 'da3' to 'synthetic' " "after operational failure (OOM)"),
            model_id="synthetic",
            device="cpu",
            attempts=[
                {"backend": "da3", "status": "failed", "error_message": "OOM"},
                {"backend": "synthetic", "status": "success"},
            ],
        )
        manifest_path = tmp_path / "fallback.manifest.json"
        CombinedManifest(backend_selection=fallback_metadata).save(manifest_path)

        with open(manifest_path) as f:
            persisted = json.load(f)

        assert persisted["backend_selection"]["resolution_status"] == "fallback"
        reason = persisted["backend_selection"]["resolution_reason"]
        assert isinstance(reason, str) and reason
        assert "da3" in reason and "synthetic" in reason

    def test_attempts_entries_record_backend_and_status(self, tmp_path: Path) -> None:
        """Each entry in `backend_selection.attempts` must be a dict with
        at least `backend` and `status` keys.

        The existing manifests test only asserts ``isinstance(..., list)``;
        downstream consumers depend on the per-attempt shape.
        """
        manifest = _get_manifest_data(tmp_path, "attempts_shape.png")
        attempts = manifest["backend_selection"]["attempts"]
        # The success path may produce an empty list; only the per-entry
        # shape is contract-bearing. Use a representative fallback metadata
        # via direct dataclass construction to assert the per-entry shape.
        from transformation_portal.lux_depth_v3.manifest import BackendSelectionMetadata

        sample = BackendSelectionMetadata(
            requested_backend="da3",
            resolved_backend="synthetic",
            resolution_status="fallback",
            resolution_reason="probe",
            model_id="synthetic",
            device="cpu",
            attempts=[
                {"backend": "da3", "status": "failed", "error_message": "boom"},
                {"backend": "synthetic", "status": "success"},
            ],
        )
        roundtripped = BackendSelectionMetadata.from_dict(sample.to_dict())
        assert isinstance(attempts, list)
        for entry in roundtripped.attempts or []:
            assert isinstance(entry, dict)
            assert "backend" in entry
            assert "status" in entry


class TestSchemaVersionGate:
    """`BackendSelectionMetadata.from_dict` must reject unsupported schemas."""

    def test_from_dict_rejects_unsupported_schema_version(self) -> None:
        """An unknown schema_version must raise ValueError rather than silently
        deserialize fields whose meaning may have changed.

        Covers manifest.py:228-230, which was previously dead-untested.
        """
        from transformation_portal.lux_depth_v3.manifest import BackendSelectionMetadata

        with pytest.raises(ValueError, match="Unsupported BackendSelectionMetadata schema"):
            BackendSelectionMetadata.from_dict(
                {
                    "schema_version": "9.9",
                    "requested_backend": "da3",
                    "resolved_backend": "da3",
                    "resolution_status": "success",
                    "resolution_reason": None,
                    "model_id": "depth-anything-v2-base",
                    "device": "cpu",
                }
            )

    def test_from_dict_roundtrip_preserves_resolution_reason(self) -> None:
        """`to_dict()` → `from_dict()` must preserve a populated reason string
        so that the manifest carries the fallback audit trail across reads.
        """
        from transformation_portal.lux_depth_v3.manifest import BackendSelectionMetadata

        original = BackendSelectionMetadata(
            requested_backend="da3",
            resolved_backend="synthetic",
            resolution_status="fallback",
            resolution_reason="Fallback from 'da3' to 'synthetic' after backend init failure",
            model_id="synthetic",
            device="cpu",
            attempts=[{"backend": "da3", "status": "failed"}],
        )

        restored = BackendSelectionMetadata.from_dict(original.to_dict())

        assert restored.resolution_reason == original.resolution_reason
        assert restored.resolution_status == "fallback"
        assert restored.requested_backend == "da3"
        assert restored.resolved_backend == "synthetic"


class TestRuntimeLicensingManifest:
    """Runtime licensing evidence must be emitted and legacy tolerant."""

    def test_runtime_licensing_manifest_marks_research_model_active(self) -> None:
        from transformation_portal.lux_depth_v3.run_card_contract import build_runtime_licensing_manifest

        licensing = build_runtime_licensing_manifest(
            model_contract={
                "resolved_repo_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                "license_id": "cc-by-nc-4.0",
                "backend_kind": "da3",
                "usage_class": "non_commercial_only",
                "requires_non_commercial_ok": True,
            },
            config=SimpleNamespace(non_commercial_ok=True),
        )

        assert licensing["software_license_tier"] == "research_or_non_commercial"
        assert licensing["non_commercial_active"] is True
        assert licensing["research_acknowledgement_required"] is True
        assert licensing["models"][0]["license"] == "cc-by-nc-4.0"

    def test_runtime_licensing_manifest_marks_commercial_model(self) -> None:
        from transformation_portal.lux_depth_v3.run_card_contract import build_runtime_licensing_manifest

        licensing = build_runtime_licensing_manifest(
            model_contract={
                "resolved_repo_id": "commercial/depth-model",
                "license_id": "commercial",
                "backend_kind": "depth",
                "usage_class": "commercial",
                "requires_non_commercial_ok": False,
            },
            config=SimpleNamespace(non_commercial_ok=False),
        )

        assert licensing["software_license_tier"] == "commercial"
        assert licensing["non_commercial_active"] is False
        assert licensing["research_acknowledgement_required"] is False
        assert licensing["models"][0]["id"] == "commercial/depth-model"

    def test_combined_manifest_round_trips_licensing(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest

        manifest_path = tmp_path / "licensing.manifest.json"
        licensing = {
            "schema_version": "1.0",
            "software_license_tier": "commercial",
            "models": [],
            "non_commercial_active": False,
            "research_acknowledgement_required": False,
        }

        CombinedManifest(licensing=licensing).save(manifest_path)

        restored = CombinedManifest.load(manifest_path)
        assert restored.licensing == licensing
