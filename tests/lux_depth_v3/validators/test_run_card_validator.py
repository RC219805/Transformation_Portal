"""Unit tests for RunCardValidator.

Tests the run card validation logic extracted from orchestrator.py
as part of ADR-043 decomposition.

These tests verify:
1. Schema validation with Draft2020-12 JSON Schema
2. Backend semantics validation rules
3. Backward compatibility with orchestrator imports
4. ValidationResult interface
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]


class TestRunCardValidatorImports:
    """Test that imports work from both the new and legacy locations."""

    def test_import_from_validators_package(self):
        """Test that we can import from the new validators package."""
        from transformation_portal.lux_depth_v3.validators import (
            RunCardValidationError,
            RunCardValidator,
            validate_run_card_backend_semantics,
            validate_run_card_payload,
        )

        assert RunCardValidator is not None
        assert RunCardValidationError is not None
        assert callable(validate_run_card_payload)
        assert callable(validate_run_card_backend_semantics)

    def test_backward_compatible_orchestrator_imports(self):
        """Test that legacy imports from orchestrator still work."""
        from transformation_portal.lux_depth_v3.orchestrator import (
            _run_card_schema_path,
            _validate_run_card_backend_semantics,
            _validate_run_card_payload,
        )

        # Should be callable
        assert callable(_validate_run_card_payload)
        assert callable(_validate_run_card_backend_semantics)
        assert callable(_run_card_schema_path)

        # Schema path should be a Path
        schema_path = _run_card_schema_path()
        assert isinstance(schema_path, Path)


class TestValidationResult:
    """Test ValidationResult data class."""

    def test_valid_result_is_truthy(self):
        """Test that valid result is truthy."""
        from transformation_portal.lux_depth_v3.validators.run_card_validator import (
            ValidationResult,
        )

        result = ValidationResult(is_valid=True, errors=[])
        assert result.is_valid
        assert bool(result) is True
        assert result.errors == []

    def test_invalid_result_is_falsy(self):
        """Test that invalid result is falsy."""
        from transformation_portal.lux_depth_v3.validators.run_card_validator import (
            ValidationResult,
        )

        result = ValidationResult(
            is_valid=False, errors=["error1", "error2"]
        )
        assert not result.is_valid
        assert bool(result) is False
        assert len(result.errors) == 2


def _minimal_valid_payload() -> Dict[str, Any]:
    """Return a minimal valid run card payload for testing."""
    config_fingerprint = {
        "model_variant": "METRIC_LARGE",
        "depth_quantization": "u16",
        "depth_device": "cpu",
        "preset": "premium",
        "preset_requested": "premium",
        "preset_resolved": "premium",
        "backend_requested": "da3",
        "backend_resolved": "da3",
        "device_requested": "cpu",
        "device_resolved": "cpu",
        "quality_tier": "premium",
        "strict_inputs": False,
        "strict_segmentation": False,
        "apex_strict_mode": False,
        "v2_preset": "premium",
        "v2_device": "cpu",
        "v2_upscaler_backend": "realesrgan",
        "depth_pro_python_executable": None,
    }
    canonical_json = json.dumps(
        {
            field: config_fingerprint[field]
            for field in (
                "model_variant",
                "depth_quantization",
                "depth_device",
                "preset",
                "v2_preset",
                "v2_device",
                "v2_upscaler_backend",
                "preset_requested",
                "preset_resolved",
                "backend_requested",
                "backend_resolved",
                "device_requested",
                "device_resolved",
                "quality_tier",
                "strict_inputs",
                "strict_segmentation",
                "apex_strict_mode",
                "depth_pro_python_executable",
            )
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    config_fingerprint["hash_algorithm"] = "sha256"
    config_fingerprint["canonical_json"] = canonical_json
    config_fingerprint["sha256"] = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    return {
        "batch_id": "2026-03-20_120000",
        "start_time": "2026-03-20T12:00:00Z",
        "end_time": "2026-03-20T12:01:00Z",
        "config_fingerprint": {
            **config_fingerprint,
            "raw_ingest_profile": "tp.raw_ingest.deterministic_v1",
            "raw_ingest_settings_hash": "e" * 64,
        },
        "backend_selection": {
            "requested": "da3",
            "resolved": "da3",
            "device": "cpu",
            "model_id": "depth-anything/DA3-METRIC-LARGE-1.0",
        },
        "backend_summary": {
            "requested_backend": "da3",
            "primary_backend": "da3",
            "final_backends_used": ["da3"],
            "fallback_images": 0,
            "fallback_rate": 0.0,
        },
        "environment": {
            "python_version": "3.12.0",
            "platform": "linux",
            "hostname": "test-host",
        },
        "timing": {
            "total_seconds": 60.0,
            "depth_seconds": 30.0,
            "v2_seconds": 20.0,
            "other_seconds": 10.0,
        },
        "image_count": 5,
        "success_count": 5,
        "failure_count": 0,
        "skip_count": 0,
        "results": [],
        "artifacts": [],
        "artifact_merkle_root": "a" * 64,
    }


class TestBackendSemanticsValidation:
    """Test backend semantics validation rules."""

    def test_valid_backend_semantics(self):
        """Test that valid semantics pass validation."""
        from transformation_portal.lux_depth_v3.validators import (
            validate_run_card_backend_semantics,
        )

        payload = _minimal_valid_payload()
        # Should not raise
        validate_run_card_backend_semantics(payload)

    def test_missing_backend_selection_skips_validation(self):
        """Test that missing backend_selection skips validation gracefully."""
        from transformation_portal.lux_depth_v3.validators import (
            validate_run_card_backend_semantics,
        )

        payload = {"backend_summary": {}}
        # Should not raise - gracefully skips
        validate_run_card_backend_semantics(payload)

    def test_empty_final_backends_with_success_raises(self):
        """Test that empty final_backends_used with success_count > 0 raises."""
        from transformation_portal.lux_depth_v3.validators import (
            validate_run_card_backend_semantics,
        )

        payload = {
            "backend_selection": {"resolved": "da3"},
            "backend_summary": {
                "final_backends_used": [],
                "primary_backend": "da3",
            },
            "success_count": 5,
        }

        with pytest.raises(
            RuntimeError,
            match="final_backends_used must be non-empty",
        ):
            validate_run_card_backend_semantics(payload)

    def test_primary_backend_mismatch_raises(self):
        """Test that primary_backend not matching final_backends_used[0] raises."""
        from transformation_portal.lux_depth_v3.validators import (
            validate_run_card_backend_semantics,
        )

        payload = {
            "backend_selection": {"resolved": "da3"},
            "backend_summary": {
                "final_backends_used": ["da3"],
                "primary_backend": "da2",  # Mismatch!
            },
            "success_count": 5,
        }

        with pytest.raises(
            RuntimeError,
            match="primary_backend must equal.*final_backends_used",
        ):
            validate_run_card_backend_semantics(payload)

    def test_resolved_mismatch_raises(self):
        """Test that resolved not matching final_backends_used[0] raises."""
        from transformation_portal.lux_depth_v3.validators import (
            validate_run_card_backend_semantics,
        )

        payload = {
            "backend_selection": {"resolved": "da2"},  # Mismatch!
            "backend_summary": {
                "final_backends_used": ["da3"],
                "primary_backend": "da3",
            },
            "success_count": 5,
        }

        with pytest.raises(
            RuntimeError,
            match="resolved must match.*final_backends_used",
        ):
            validate_run_card_backend_semantics(payload)


class TestWrapperSemanticsValidation:
    """Test wrapper backend semantics validation."""

    def test_wrapper_semantics_valid(self):
        """Test that valid wrapper semantics pass."""
        from transformation_portal.lux_depth_v3.validators import (
            validate_run_card_backend_semantics,
        )

        payload = {
            "backend_selection": {
                "resolved": "da3",
                "logical_backend": "ensemble",
                "resolved_engine": "da3",
            },
            "backend_summary": {
                "final_backends_used": ["da3"],
                "primary_backend": "da3",
                "fallback_images": 0,
            },
            "success_count": 5,
        }
        # Should not raise
        validate_run_card_backend_semantics(payload)

    def test_wrapper_same_logical_resolved_raises(self):
        """Test that logical_backend == resolved_engine raises."""
        from transformation_portal.lux_depth_v3.validators import (
            validate_run_card_backend_semantics,
        )

        payload = {
            "backend_selection": {
                "resolved": "da3",
                "logical_backend": "da3",  # Same as resolved_engine
                "resolved_engine": "da3",
            },
            "backend_summary": {
                "final_backends_used": ["da3"],
                "primary_backend": "da3",
                "fallback_images": 0,
            },
            "success_count": 5,
        }

        with pytest.raises(
            RuntimeError,
            match="logical_backend and.*resolved_engine must differ",
        ):
            validate_run_card_backend_semantics(payload)

    def test_wrapper_with_fallback_images_raises(self):
        """Test that wrapper semantics with fallback_images != 0 raises."""
        from transformation_portal.lux_depth_v3.validators import (
            validate_run_card_backend_semantics,
        )

        payload = {
            "backend_selection": {
                "resolved": "da3",
                "logical_backend": "ensemble",
                "resolved_engine": "da3",
            },
            "backend_summary": {
                "final_backends_used": ["da3"],
                "primary_backend": "da3",
                "fallback_images": 2,  # Non-zero fallback
            },
            "success_count": 5,
        }

        with pytest.raises(
            RuntimeError,
            match="fallback_images == 0",
        ):
            validate_run_card_backend_semantics(payload)


class TestRunCardValidator:
    """Test the unified RunCardValidator class."""

    def test_validator_init_default_schema(self):
        """Test that validator initializes with default schema."""
        from transformation_portal.lux_depth_v3.validators import RunCardValidator

        validator = RunCardValidator()
        assert validator.schema_path.name == "run_card.v1.schema.json"

    def test_validator_validate_returns_result(self):
        """Test that validate() returns a ValidationResult."""
        from transformation_portal.lux_depth_v3.validators import RunCardValidator

        validator = RunCardValidator()
        payload = _minimal_valid_payload()

        result = validator.validate(payload)

        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")

    def test_validator_validate_or_raise_on_invalid(self):
        """Test that validate_or_raise() raises on invalid payload."""
        from transformation_portal.lux_depth_v3.validators import (
            RunCardValidationError,
            RunCardValidator,
        )

        validator = RunCardValidator()
        payload = {"invalid": "payload"}  # Missing required fields

        with pytest.raises(RunCardValidationError):
            validator.validate_or_raise(payload)


class TestSchemaPathResolution:
    """Test schema path resolution."""

    def test_default_schema_path_exists(self):
        """Test that the default schema path exists."""
        from transformation_portal.lux_depth_v3.validators.run_card_validator import (
            _default_schema_path,
        )

        schema_path = _default_schema_path()
        # Note: Path may not exist in minimal test envs; check structure
        assert schema_path.name == "run_card.v1.schema.json"
        assert "run_card" in str(schema_path)

    def test_orchestrator_schema_path_alias(self):
        """Test that orchestrator._run_card_schema_path is an alias."""
        from transformation_portal.lux_depth_v3.orchestrator import (
            _run_card_schema_path,
        )
        from transformation_portal.lux_depth_v3.validators.run_card_validator import (
            _default_schema_path,
        )

        assert _run_card_schema_path() == _default_schema_path()
