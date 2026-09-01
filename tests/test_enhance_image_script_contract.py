"""Contract tests for the standalone V2 enhancement script naming behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "enhance_image.py"
RAW_SUFFIXES = (".dng", ".cr2", ".nef", ".arw")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("tp_v2_enhance_script_contract", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_input_fixture(path: Path) -> None:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        data = np.full((8, 8, 3), 96, dtype=np.uint8)
        Image.fromarray(data, mode="RGB").save(path)
        return

    path.write_bytes(b"raw-input-fixture")


@pytest.mark.parametrize(
    ("input_name", "asset_key", "expected_name"),
    [
        ("scene.DNG", None, "scene_v2_enhanced.tif"),
        ("scene.CR2", "scene_hash", "scene_hash_v2_enhanced.tif"),
        ("scene.jpg", None, "scene_v2_enhanced.png"),
        ("scene.png", "scene_hash", "scene_hash_v2_enhanced.png"),
    ],
)
def test_run_v2_enhancement_uses_canonical_emitted_artifact_name(
    tmp_path: Path,
    input_name: str,
    asset_key: str | None,
    expected_name: str,
) -> None:
    module = _load_script_module()

    input_path = tmp_path / input_name
    output_dir = tmp_path / "output"
    _write_input_fixture(input_path)

    def _fake_enhance_image(
        *,
        input_path: Path,
        output_path: Path,
        depth_map_path: Path | None,
        material_masks: dict[str, Any] | None,
        config: Any,
        device: str,
        allow_8bit_output: bool,
        output_bit_depth: int | None,
    ) -> dict[str, Any]:
        assert output_bit_depth is None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"enhanced")
        return {
            "status": "success",
            "implementation": "v2_enhance",
            "input": str(input_path),
            "output": str(output_path),
            "depth_map": str(depth_map_path) if depth_map_path else None,
            "depth_consumed": False,
            "preset": config.preset,
            "runtime_s": 0.01,
            "timestamp": 0.0,
            "depth": {
                "requested": False,
                "resolved_path": None,
                "loaded": False,
                "supplied_to_stage": False,
                "consumed": False,
                "consumption_source": "not_requested",
                "stage_has_depth": None,
            },
        }

    with patch.object(module, "enhance_image", side_effect=_fake_enhance_image):
        report = module.run_v2_enhancement(
            input_path=input_path,
            depth_dir=None,
            output_dir=output_dir,
            preset="default",
            device="cpu",
            upscaler="default",
            allow_8bit=False,
            masks_file=None,
            asset_key=asset_key,
        )

    emitted_output = Path(report["output"])
    assert emitted_output.name == expected_name
    assert emitted_output.parent == output_dir
    assert report["asset_key"] == (asset_key or input_path.stem)
    assert report["input_stem"] == input_path.stem
    assert emitted_output.exists()
    assert all(not report["output"].lower().endswith(raw_suffix) for raw_suffix in RAW_SUFFIXES)
