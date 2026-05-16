"""Orchestrator-level input-discovery wiring tests.

`tests/test_input_discovery.py` already exercises `discover_images()`
exhaustively at the unit level. The integration call site —
`enhance_batch(input_dir=...)` at `orchestrator.py:4946-4954`, which
builds a `DiscoveryConfig` from `self.config.strict_inputs` and passes
`self.output_root` as the exclusion — had no test mirror.

These tests close the depth-of-depth contract gap CLAUDE.md calls out:

    "Input discovery deliberately excludes derived artifacts and output
    dirs to prevent 'depth-of-depth' loops — do not weaken this filter."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def _make_test_image(tmp_path: Path, name: str, size: tuple = (64, 64)) -> Path:
    """Create a minimal RGB test image."""
    image_path = tmp_path / name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(image_path)
    return image_path


def _make_mock_depth_result():
    """Create a deterministic synthetic depth result."""
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
    """Create a mock depth backend registry."""
    backend = Mock()
    backend.name = "da3"
    backend.license_type = Mock(value="commercial")
    backend.ensure_available.return_value = None
    backend.compute.return_value = _make_mock_depth_result()

    registry = Mock()
    registry.get_backend.return_value = backend
    return registry


def _create_orchestrator(tmp_path: Path, output_root: Path = None, **config_kwargs):
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
        orchestrator = EnhanceOrchestrator(config, output_root or tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        return orchestrator


class _DiscoverImagesCapture:
    """Spy that records the kwargs passed to `discover_images` and returns
    an empty list to short-circuit downstream batch processing."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, input_dir, config, image_extensions=None, output_dir=None):
        self.calls.append(
            {
                "input_dir": input_dir,
                "config": config,
                "image_extensions": image_extensions,
                "output_dir": output_dir,
            }
        )
        return []


class TestEnhanceBatchWiresOutputRoot:
    """`enhance_batch(input_dir)` must pass `self.output_root` as the
    `output_dir` exclusion so prior runs cannot loop into themselves."""

    def test_output_root_is_passed_as_discover_output_dir(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_root = tmp_path / "input" / "output"
        input_dir.mkdir()
        output_root.mkdir(parents=True)

        orchestrator = _create_orchestrator(tmp_path, output_root=output_root)
        spy = _DiscoverImagesCapture()
        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.discover_images",
            spy,
        ):
            orchestrator.enhance_batch(input_dir=input_dir)

        assert len(spy.calls) == 1
        # `output_dir` may be the literal output_root or its resolved real
        # path; assert by name comparison rather than identity.
        captured_output = spy.calls[0]["output_dir"]
        assert Path(captured_output) == output_root
        assert Path(spy.calls[0]["input_dir"]) == input_dir


class TestEnhanceBatchPropagatesStrictInputs:
    """The `strict_inputs` config flag must flow into the DiscoveryConfig."""

    @pytest.mark.parametrize("strict", [True, False])
    def test_strict_inputs_flag_threaded_through(self, tmp_path: Path, strict: bool) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        orchestrator = _create_orchestrator(tmp_path, strict_inputs=strict)
        spy = _DiscoverImagesCapture()
        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.discover_images",
            spy,
        ):
            orchestrator.enhance_batch(input_dir=input_dir)

        assert spy.calls[0]["config"].strict_mode is strict


class TestEnhanceBatchDefaultExtensions:
    """The Path-taking `enhance_batch` overload defaults image_extensions to
    the standard set declared at `orchestrator.py:4908-4909`."""

    def test_default_extensions_are_standard_set(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        orchestrator = _create_orchestrator(tmp_path)
        spy = _DiscoverImagesCapture()
        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.discover_images",
            spy,
        ):
            orchestrator.enhance_batch(input_dir=input_dir)

        assert spy.calls[0]["image_extensions"] == [
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
        ]


class TestEnhanceBatchDoesNotReingestDepthArtifacts:
    """End-to-end: a `*_depth16.png` artifact sharing the input directory
    must not be re-ingested as a fresh source.

    This is the "no depth-of-depth" loop guard at the orchestrator boundary
    — proven here by counting how many entries reach the result list, not
    by inspecting filter internals (which `tests/test_input_discovery.py`
    already does)."""

    def test_depth_sibling_is_filtered_out(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_root = tmp_path / "outputs"
        input_dir.mkdir()
        output_root.mkdir()

        # One clean RGB plus one *_depthpro_depth16.png sibling in the
        # same dir. The stem `villa_depthpro_depth16` matches the
        # `_depthpro_depth16` entry in DiscoveryConfig.exclude_stem_suffixes
        # (see input_discovery.py:60), so the artifact must be skipped.
        _make_test_image(input_dir, "villa.png")
        depth_artifact = input_dir / "villa_depthpro_depth16.png"
        Image.fromarray(
            np.zeros((64, 64), dtype=np.uint16),
            mode="I;16",
        ).save(depth_artifact)

        orchestrator = _create_orchestrator(tmp_path, output_root=output_root)
        results = orchestrator.enhance_batch(input_dir=input_dir)

        # Only the clean RGB should be processed; the depth artifact must
        # be filtered out by the stem-suffix exclusion.
        assert isinstance(results, list)
        assert len(results) == 1
        processed_input = results[0].get("input_path") or results[0].get("input")
        if processed_input is not None:
            assert "depthpro_depth16" not in str(processed_input)
