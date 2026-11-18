"""Smoke coverage for the 4K rendering pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.unit

PIPELINE_MODULE = "transformation_portal.pipelines.rendering_4k_pipeline"
CONTROLNET_AUX_MODULE = "controlnet_aux"


def _load_rendering_4k_pipeline_for_preview_smoke(monkeypatch):
    # The preview preset is expected to keep AI enhancement disabled.
    # Force optional ControlNet support unavailable as well so this smoke test
    # stays deterministic even when extra local dependencies are installed.
    monkeypatch.setitem(sys.modules, CONTROLNET_AUX_MODULE, None)
    monkeypatch.delitem(sys.modules, PIPELINE_MODULE, raising=False)

    return importlib.import_module(PIPELINE_MODULE)


def test_rendering_4k_pipeline_preview_smoke_writes_delivery_artifact(tmp_path: Path, monkeypatch) -> None:
    pipeline_module = _load_rendering_4k_pipeline_for_preview_smoke(monkeypatch)

    try:
        input_path = tmp_path / "input.png"
        output_dir = tmp_path / "output"
        Image.new("RGB", (8, 6), color=(64, 96, 128)).save(input_path)

        pipeline = pipeline_module.Rendering4KPipeline.from_preset("preview")
        assert pipeline.config.name == "preview"
        assert not pipeline.config.ai_enhancement.enabled

        result = pipeline.process(input_path, output_dir)

        assert result.config_used is not None
        assert result.config_used.name == "preview"
        assert result.depth_map is None
        assert result.iterations == 1
        assert result.image.mode == "RGB"
        assert result.image.size == (8, 6)
        assert result.output_paths.keys() == {"delivery_jpeg"}

        output_path = result.output_paths["delivery_jpeg"]
        assert output_path == output_dir / "input_DELIVERY.jpg"
        assert output_path.exists()

        with Image.open(output_path) as output_image:
            assert output_image.mode == "RGB"
            assert output_image.size == (8, 6)
    finally:
        sys.modules.pop(PIPELINE_MODULE, None)


def test_rendering_4k_depth_cache_does_not_alias_images_sharing_a_prefix() -> None:
    """M-01: two frames that differ only after the first rows must not share a depth cache entry."""
    import numpy as np

    from transformation_portal.pipelines.rendering_4k import stages
    from transformation_portal.pipelines.rendering_4k.pipeline import Rendering4KPipeline

    pipeline = Rendering4KPipeline.from_preset("default")
    assert pipeline.config.depth.cache_enabled
    # Force the deterministic fallback estimator; no model is loaded.
    pipeline._depth_model_initialized = True
    pipeline._depth_model = None

    image_a = np.zeros((128, 512, 3), np.float32)
    image_a[:1] = 1.0
    image_a[1:, :256] = 0.9
    image_b = np.zeros((128, 512, 3), np.float32)
    image_b[:1] = 1.0
    image_b[1:, 256:] = 0.9

    assert pipeline._compute_cache_key(image_a) != pipeline._compute_cache_key(image_b)

    depth_a = pipeline._estimate_depth(image_a, Path("a.png")).copy()
    depth_b = pipeline._estimate_depth(image_b, Path("b.png"))

    np.testing.assert_array_equal(depth_b, stages.estimate_depth_simple(image_b))
    assert not np.array_equal(depth_a, depth_b)
    # Identical content still hits the cache.
    assert (
        pipeline._estimate_depth(image_a.copy(), Path("a2.png")) is pipeline._depth_cache[pipeline._compute_cache_key(image_a)]
    )


def _write_tagged_jpeg(path: Path) -> bytes:
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    img = Image.new("RGB", (8, 6), color=(64, 96, 128))
    exif = img.getexif()
    exif[0x0110] = "Camera"
    img.save(path, icc_profile=icc, dpi=(300, 300), exif=exif, quality=95)
    return icc


@pytest.mark.parametrize("preserve", [True, False])
def test_rendering_4k_outputs_carry_source_icc_and_dpi(tmp_path: Path, monkeypatch, preserve: bool) -> None:
    """M-07: preserve_metadata=True re-attaches the source ICC profile and DPI to the outputs."""
    pytest.importorskip("tifffile")
    pipeline_module = _load_rendering_4k_pipeline_for_preview_smoke(monkeypatch)

    try:
        input_path = tmp_path / "tagged.jpg"
        icc = _write_tagged_jpeg(input_path)

        pipeline = pipeline_module.Rendering4KPipeline.from_preset("preview")
        pipeline.config.output.master_tiff_16bit = True
        pipeline.config.output.preserve_metadata = preserve
        result = pipeline.process(input_path, tmp_path / "output")

        assert {"master_tiff", "delivery_jpeg"} <= result.output_paths.keys()
        for key in ("master_tiff", "delivery_jpeg"):
            with Image.open(result.output_paths[key]) as output_image:
                if preserve:
                    assert output_image.info.get("icc_profile") == icc
                    assert tuple(round(v) for v in output_image.info["dpi"]) == (300, 300)
                else:
                    assert not output_image.info.get("icc_profile")
        with Image.open(result.output_paths["delivery_jpeg"]) as jpeg:
            assert (jpeg.getexif().get(0x0110) == "Camera") is preserve
    finally:
        sys.modules.pop(PIPELINE_MODULE, None)


def test_rendering_4k_cli_batch_exits_nonzero_when_every_image_fails(tmp_path: Path, monkeypatch) -> None:
    """M-08: a batch in which every image failed must not exit 0."""
    pipeline_module = _load_rendering_4k_pipeline_for_preview_smoke(monkeypatch)

    try:
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "a.jpg").write_bytes(b"not an image")
        (bad_dir / "b.png").write_bytes(b"also not an image")
        monkeypatch.setattr(
            sys,
            "argv",
            ["rendering_4k_pipeline", "-d", str(bad_dir), "-o", str(tmp_path / "out"), "--preset", "preview"],
        )

        assert pipeline_module.main() == 1
    finally:
        sys.modules.pop(PIPELINE_MODULE, None)


def test_rendering_4k_batch_same_stem_inputs_keep_separate_deliveries(tmp_path: Path, monkeypatch) -> None:
    """M-02: kitchen.png and kitchen.tif must not write the same _DELIVERY.jpg."""
    import numpy as np

    pipeline_module = _load_rendering_4k_pipeline_for_preview_smoke(monkeypatch)

    try:
        bright = tmp_path / "kitchen.png"
        dark = tmp_path / "kitchen.tif"
        Image.new("RGB", (8, 6), color=(250, 250, 250)).save(bright)
        Image.new("RGB", (8, 6), color=(10, 10, 10)).save(dark)
        output_dir = tmp_path / "output"

        pipeline = pipeline_module.Rendering4KPipeline.from_preset("preview")
        results = pipeline.batch_process([bright, dark], output_dir, show_progress=False)

        paths = [r.output_paths["delivery_jpeg"] for r in results]
        assert paths == [output_dir / "kitchen_DELIVERY.jpg", output_dir / "kitchen_tif_DELIVERY.jpg"]
        means = [float(np.asarray(Image.open(p)).mean()) for p in paths]
        assert means[0] > means[1] + 100
    finally:
        sys.modules.pop(PIPELINE_MODULE, None)
