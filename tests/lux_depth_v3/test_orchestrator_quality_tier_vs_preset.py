"""Tests proving `quality_tier` and `--preset` remain distinct concepts.

CLAUDE.md is explicit:

    "Quality tier (`standard|premium|apex`) and `--preset` are **distinct**
    concepts."

Existing tests partially cover the pair: `test_config_resolver.py:718`
asserts both keys appear in the fingerprint, and
`test_pipeline_coordinator.py:769-780` confirms `quality_tier` flows to
the plan. Neither test asserts that the two values stay independent
across the full cross-product of preset × tier, or that the orchestrator
persists both into the run-card fingerprint without one silently
coercing the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def _make_test_image(tmp_path: Path, name: str = "test.png", size: tuple = (64, 64)) -> Path:
    image_path = tmp_path / name
    Image.new("RGB", size, color="white").save(image_path)
    return image_path


def _make_mock_depth_result():
    from transformation_portal.depth.backends.protocol import DepthResult

    return DepthResult(
        depth_map=np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        metadata={},
        depth_units="relative",
        backend_id="da3",
        device="cpu",
    )


def _make_mock_registry():
    backend = Mock()
    backend.name = "da3"
    backend.license_type = Mock(value="commercial")
    backend.ensure_available.return_value = None
    backend.compute.return_value = _make_mock_depth_result()

    registry = Mock()
    registry.get_backend.return_value = backend
    return registry


def _create_orchestrator(tmp_path: Path, **config_kwargs):
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


def _get_manifest_data(tmp_path: Path, image_name: str, **config_kwargs) -> Dict[str, Any]:
    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    orchestrator = _create_orchestrator(tmp_path, **config_kwargs)
    test_image = _make_test_image(tmp_path, image_name)
    result = orchestrator.enhance_image(
        ImageInput(path=test_image),
        input_root=tmp_path,
    )
    manifest_path = Path(result["manifest"])
    with open(manifest_path) as f:
        return json.load(f)


class TestPresetTierFingerprintCrossProduct:
    """Every (preset, quality_tier) cell must yield independent fingerprint
    keys — neither value may shadow or coerce the other."""

    @pytest.mark.parametrize("tier", ["standard", "premium", "apex"])
    @pytest.mark.parametrize(
        "preset_member",
        ["DEFAULT", "LUXURY_ESTATE"],
    )
    def test_preset_and_tier_flow_independently_into_fingerprint(self, preset_member: str, tier: str) -> None:
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, Preset
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_run_card_config_fingerprint,
        )

        preset = getattr(Preset, preset_member)
        config = EnhanceConfig(preset=preset, quality_tier=tier)
        fingerprint = build_run_card_config_fingerprint(config)

        # With `config.preset` set, `config_resolver.py:610-611` always
        # resolves both `preset_requested` and `preset_resolved` to
        # `preset.value` — never None. The cross-product must hold for
        # every (preset, tier) cell.
        assert fingerprint["preset_requested"] == preset.value
        assert fingerprint["preset_resolved"] == preset.value
        # quality_tier survives untouched regardless of which preset is set.
        assert fingerprint["quality_tier"] == tier


class TestNoPresetFallback:
    """When `preset` is None, `preset_resolved` falls back to a
    `quality_tier:<tier>` marker; the tier itself must not be rewritten."""

    @pytest.mark.parametrize("tier", ["standard", "premium", "apex"])
    def test_no_preset_resolves_to_quality_tier_marker(self, tier: str) -> None:
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_run_card_config_fingerprint,
        )

        config = EnhanceConfig(preset=None, quality_tier=tier)
        fingerprint = build_run_card_config_fingerprint(config)

        assert fingerprint["preset_requested"] is None
        assert fingerprint["preset_resolved"] == f"quality_tier:{tier}"
        # The tier-as-marker fallback must not mutate the tier field.
        assert fingerprint["quality_tier"] == tier


class TestQualityTierDoesNotForcePresetUpgrade:
    """`quality_tier='apex'` must NOT silently promote a low-end preset.

    This guards against a refactor regression where apex-mode validation
    rewrites the preset to LUXURY_ESTATE behind the user's back.
    """

    def test_apex_tier_preserves_default_preset(self) -> None:
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, Preset
        from transformation_portal.lux_depth_v3.config_resolver import (
            build_run_card_config_fingerprint,
        )

        config = EnhanceConfig(preset=Preset.DEFAULT, quality_tier="apex")
        fingerprint = build_run_card_config_fingerprint(config)

        assert fingerprint["preset_resolved"] == "default"
        assert fingerprint["quality_tier"] == "apex"


class TestOrchestratorPersistsBothInRunCard:
    """The orchestrator's run-card fingerprint helper must carry both
    `preset_requested` and `quality_tier` independently.

    The manifest's `ConfigFingerprint` (compute_config_fingerprint) only
    carries `preset`; the richer run-card payload from
    `_build_run_card_config_fingerprint` is the contract surface that the
    run-card consumer reads. We exercise the orchestrator's wrapper
    method directly so the test does not depend on run-card file emission
    timing or path conventions.
    """

    def test_run_card_fingerprint_carries_both_preset_and_tier(self, tmp_path: Path) -> None:
        from transformation_portal.lux_depth_v3.config import Preset

        orchestrator = _create_orchestrator(
            tmp_path,
            preset=Preset.LUXURY_ESTATE,
            quality_tier="apex",
        )

        fingerprint = orchestrator._build_run_card_config_fingerprint()

        assert fingerprint["preset_requested"] == "luxury_estate"
        assert fingerprint["preset_resolved"] == "luxury_estate"
        assert fingerprint["quality_tier"] == "apex"
