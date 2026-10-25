from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "ad_editorial_post_pipeline.py"


def _load_tool(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(sys.modules, "rawpy", ModuleType("rawpy"))
    reportlab_module = ModuleType("reportlab")
    reportlab_lib_module = ModuleType("reportlab.lib")
    reportlab_pagesizes_module = ModuleType("reportlab.lib.pagesizes")
    reportlab_pagesizes_module.A4 = (595.27, 841.89)
    reportlab_utils_module = ModuleType("reportlab.lib.utils")
    reportlab_utils_module.ImageReader = object
    reportlab_pdfgen_module = ModuleType("reportlab.pdfgen")
    reportlab_canvas_module = ModuleType("reportlab.pdfgen.canvas")

    class _Canvas:
        def __init__(self, *args, **kwargs) -> None:
            pass

    reportlab_canvas_module.Canvas = _Canvas
    monkeypatch.setitem(sys.modules, "reportlab", reportlab_module)
    monkeypatch.setitem(sys.modules, "reportlab.lib", reportlab_lib_module)
    monkeypatch.setitem(sys.modules, "reportlab.lib.pagesizes", reportlab_pagesizes_module)
    monkeypatch.setitem(sys.modules, "reportlab.lib.utils", reportlab_utils_module)
    monkeypatch.setitem(sys.modules, "reportlab.pdfgen", reportlab_pdfgen_module)
    monkeypatch.setitem(sys.modules, "reportlab.pdfgen.canvas", reportlab_canvas_module)

    spec = importlib.util.spec_from_file_location("ad_editorial_post_pipeline_under_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_exports_work_in_isolated_editorial_python_without_installed_project() -> None:
    # This exercises the runtime's -I source loading, not native RAW/PDF
    # acceptance. Those optional imports are stubs as in the focused fixture.
    script = r"""
import importlib.abc, importlib.util, io, sys, types
class NoInstalledProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "transformation_portal" or fullname.startswith("transformation_portal."):
            raise ModuleNotFoundError("The governed editorial closure does not install the project")
sys.meta_path.insert(0, NoInstalledProject())
for name in ("rawpy", "reportlab", "reportlab.lib", "reportlab.lib.pagesizes",
             "reportlab.lib.utils", "reportlab.pdfgen", "reportlab.pdfgen.canvas"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["reportlab.lib.pagesizes"].A4 = (595.27, 841.89)
sys.modules["reportlab.lib.utils"].ImageReader = object
sys.modules["reportlab.pdfgen.canvas"].Canvas = object
spec = importlib.util.spec_from_file_location("editorial_isolated_contract", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
import numpy as np
from PIL import Image
samples = np.linspace(.001, .003, 12 * 18 * 3, dtype=np.float32).reshape(12, 18, 3)
resized = module.resize_long_edge(samples, 9)
assert resized.shape == (6, 9, 3) and np.unique(resized).size > 100
encoded = module.jpeg_bytes(resized, icc_profile=module._srgb_icc())
with Image.open(io.BytesIO(encoded)) as image:
    assert image.size == (9, 6) and image.info["icc_profile"]
assert not any(name == "transformation_portal" or name.startswith("transformation_portal.") for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(TOOL_PATH)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def _cfg(module: ModuleType, project_root: Path, *, selects: dict | None = None, metadata: dict | None = None):
    input_raw_dir = project_root / "input_raw"
    input_raw_dir.mkdir(parents=True, exist_ok=True)
    return module.PipelineConfig(
        project_name="SmithResidence",
        project_root=project_root,
        input_raw_dir=input_raw_dir,
        backup_raw_dir=None,
        rename={"enabled": False},
        selects=selects or {"use_csv": True},
        icc={},
        processing={"workers": 1},
        styles={"natural": {"exposure": 0, "contrast": 0, "saturation": 0}},
        consistency={"target_median": 0.42, "wb_neutralize": True},
        retouch={"dust_remove": False, "hotspot_reduce": False},
        export={
            "web_long_edge_px": 2500,
            "jpeg_quality": 96,
            "sharpen_web_amount": 0.35,
            "sharpen_print_amount": 0.1,
        },
        metadata=metadata or {},
        deliver={"zip": False},
    )


def test_selects_csv_relative_config_path_resolves_under_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool(monkeypatch)
    project_root = tmp_path / "project"
    cfg = _cfg(module, project_root, selects={"use_csv": True, "csv_path": "DOCS/selects.csv"})
    layout = module.Layout.build(cfg)

    csv_path = module.ensure_selects_csv(cfg, layout, [project_root / "RAW" / "Originals" / "image.CR3"])

    assert csv_path == project_root / "DOCS" / "selects.csv"
    assert csv_path.exists()
    assert not (PROJECT_ROOT / "DOCS" / "selects.csv").exists()


def test_selects_csv_absolute_config_path_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool(monkeypatch)
    project_root = tmp_path / "project"
    external_selects = tmp_path / "selects" / "keepers.csv"
    cfg = _cfg(module, project_root, selects={"use_csv": True, "csv_path": str(external_selects)})
    layout = module.Layout.build(cfg)

    csv_path = module.ensure_selects_csv(cfg, layout, [project_root / "RAW" / "Originals" / "image.CR3"])

    assert csv_path == external_selects
    assert external_selects.exists()


def test_filter_selects_reads_relative_config_path_from_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool(monkeypatch)
    project_root = tmp_path / "project"
    cfg = _cfg(module, project_root, selects={"use_csv": True, "csv_path": "DOCS/selects.csv"})
    selects_path = project_root / "DOCS" / "selects.csv"
    selects_path.parent.mkdir(parents=True)
    selects_path.write_text("filename,keep,notes\nkeep.CR3,1,\nskip.CR3,0,\n", encoding="utf-8")

    files = [project_root / "RAW" / "Originals" / "keep.CR3", project_root / "RAW" / "Originals" / "skip.CR3"]

    assert module.filter_selects(cfg, files) == [files[0]]


def test_metadata_csv_relative_config_path_resolves_under_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool(monkeypatch)
    project_root = tmp_path / "project"
    cfg = _cfg(module, project_root, metadata={"csv_path": "DOCS/metadata.csv"})

    metadata_path = module.project_relative_path(
        cfg.project_root,
        cfg.metadata.get("csv_path"),
        cfg.project_root / "DOCS" / "metadata.csv",
    )

    assert metadata_path == project_root / "DOCS" / "metadata.csv"


def test_tiff_output_names_use_canonical_tif_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_tool(monkeypatch)

    assert module.TIFF_SUFFIX == ".tif"
    assert module.tiff_filename("image") == "image.tif"
    assert module.tiff_filename("image", "_HDR") == "image_HDR.tif"
    assert module.tiff_filename("image", "_PANO") == "image_PANO.tif"


def test_active_pipeline_source_no_longer_uses_truncated_ti_suffix() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")

    assert '".ti"' not in source
    assert '"*.ti"' not in source


def test_contact_sheet_names_use_pdf_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_tool(monkeypatch)

    assert module.PDF_SUFFIX == ".pdf"
    assert module.contact_sheet_filename("natural") == "contact_natural.pdf"
    assert Path(module.contact_sheet_filename("natural")).suffix != ".pd"


def test_load_image_float_preserves_uint16_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_tool(monkeypatch)
    image = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    path = tmp_path / "intermediate.tif"
    module.tifffile.imwrite(path, image, photometric="rgb")

    result = module.load_image_float(path)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.array([[[0.0, 32768 / 65535.0, 1.0]]], dtype=np.float32))


def test_load_image_float_preserves_uint8_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_tool(monkeypatch)
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    path = tmp_path / "preview.png"
    module.Image.fromarray(image).save(path)

    result = module.load_image_float(path)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.array([[[0.0, 128 / 255.0, 1.0]]], dtype=np.float32))


def test_active_pipeline_intermediate_reloads_use_bit_depth_aware_loader() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")

    assert "np.array(Image.open(p)).astype(np.float32) / 255.0" not in source
    assert "np.array(Image.open(pt)).astype(np.float32) / 255.0" not in source
    assert "load_image_float(p)" in source
    assert "load_image_float(pt)" in source


@pytest.fixture
def editorial(monkeypatch):
    return _load_tool(monkeypatch)


def test_rgb16_tiff_round_trip_and_independent_reader(editorial, tmp_path):
    pixels = np.linspace(0, 1, 4096 * 3, dtype=np.float32).reshape((32, 128, 3))
    destination = tmp_path / "master.tif"
    editorial.save_tiff16_prophoto(pixels, destination, None)
    with editorial.tifffile.TiffFile(destination) as tif:
        assert tif.pages[0].bitspersample == 16
        assert tif.pages[0].photometric.name == "RGB"
        assert tif.pages[0].tags[34675].value == editorial.linear_prophoto_icc()
    cv2 = pytest.importorskip("cv2")
    independent = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)[..., ::-1]
    assert independent.dtype == np.uint16
    np.testing.assert_array_equal(independent, np.rint(pixels.astype(np.float64) * 65535).astype(np.uint16))
    assert np.unique(independent).size > 256
    np.testing.assert_allclose(editorial.load_image_float(destination), pixels, atol=0.5 / 65535 + 1e-7)


def _gamma_prophoto_profile(editorial):
    """Independent ICC test fixture: append a shared u8Fixed8 gamma curve."""
    import struct

    profile = bytearray(editorial.linear_prophoto_icc())
    offset = len(profile)
    curve = b"curv" + bytes(4) + struct.pack(">IH", 1, round(1.8 * 256))
    count = struct.unpack_from(">I", profile, 128)[0]
    for index in range(count):
        record = 132 + 12 * index
        if profile[record : record + 4] in (b"rTRC", b"gTRC", b"bTRC"):
            struct.pack_into(">II", profile, record + 4, offset, len(curve))
    profile += curve + bytes((-len(curve)) % 4)
    struct.pack_into(">I", profile, 0, len(profile))
    return bytes(profile)


def test_tiff_matches_supplied_profile_transfer(editorial, tmp_path):
    pixels = np.linspace(0, 1, 3000, dtype=np.float32).reshape((20, 50, 3))
    profile = _gamma_prophoto_profile(editorial)
    path = tmp_path / "encoded-prophoto.tif"
    editorial.save_tiff16_prophoto(pixels, path, profile)
    stored = editorial.tifffile.imread(path)
    gamma = round(1.8 * 256) / 256
    expected = np.rint(pixels.astype(np.float64) ** (1 / gamma) * 65535)
    np.testing.assert_allclose(stored, expected, atol=1)
    np.testing.assert_allclose(editorial.load_image_float(path), pixels, atol=1.8 / 65535)
    with editorial.tifffile.TiffFile(path) as tif:
        assert tif.pages[0].tags[34675].value == profile


def test_web_conversion_matches_independent_littlecms(editorial):
    import io

    from PIL import Image, ImageCms

    # Linear source ICC lets LittleCMS supply independent chromatic adaptation,
    # RGB matrices, and destination transfer; only its 8-bit test input is quantized.
    samples = np.array([[[46, 46, 46], [90, 55, 30], [30, 80, 45], [20, 40, 90]]], dtype=np.uint8)
    reference = ImageCms.profileToProfile(
        Image.fromarray(samples),
        ImageCms.ImageCmsProfile(io.BytesIO(editorial.linear_prophoto_icc())),
        ImageCms.createProfile("sRGB"),
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )
    actual = editorial.linear_prophoto_to_srgb(samples.astype(np.float32) / 255)
    np.testing.assert_allclose(actual, np.asarray(reference) / 255, atol=2 / 255)
    gray = editorial.linear_prophoto_to_srgb(np.full((1, 1, 3), 0.18, dtype=np.float32))
    np.testing.assert_allclose(gray, 0.4613561295, atol=1e-6)


def test_tiff_atomic_failure_preserves_existing_destination(editorial, monkeypatch, tmp_path):
    path = tmp_path / "master.tif"
    path.write_bytes(b"previous successful output")

    def fail_after_partial_write(temporary, *args, **kwargs):
        Path(temporary).write_bytes(b"partial")
        raise OSError("simulated encoder failure")

    monkeypatch.setattr(editorial.tifffile, "imwrite", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated encoder failure"):
        editorial.save_tiff16_prophoto(np.zeros((3, 4, 3)), path, None)
    assert path.read_bytes() == b"previous successful output"
    assert not path.with_suffix(".tif.tmp").exists()


def test_rgb_writer_rejects_wrong_profile_and_nonfinite_before_publication(editorial, tmp_path):
    from PIL import ImageCms

    path = tmp_path / "master.tif"
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    with pytest.raises(ValueError, match="primaries"):
        editorial.save_tiff16_prophoto(np.zeros((3, 4, 3)), path, srgb)
    with pytest.raises(ValueError, match="finite"):
        editorial.save_tiff16_prophoto(np.full((3, 4, 3), np.nan), path, None)
    assert not path.exists()
    assert not path.with_suffix(".tif.tmp").exists()


def test_float_grading_resize_and_sharpen_retain_sub_8bit_detail(editorial):
    import colorsys

    ramp = np.linspace(0.2, 0.3, 2048, dtype=np.float32)
    image = np.stack([ramp, ramp * 0.8, ramp * 0.6], axis=-1)[None, ...]
    hsv = editorial._rgb_to_hsv(image)
    reference = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in image[0]])
    np.testing.assert_allclose(hsv[0] / [360, 1, 1], reference, atol=1e-6)
    np.testing.assert_allclose(editorial._hsv_to_rgb(hsv), image, atol=1e-7)
    for output in (
        editorial.adjust_saturation(image, 5),
        editorial.split_tone(image, 210, 0.04, None, 0),
        editorial.resize_long_edge(image, 1024),
        editorial.unsharp_mask(image),
    ):
        assert np.isfinite(output).all()
        assert np.unique(output[..., 0]).size > 256
    constant = np.full((13, 17, 3), 0.12345, dtype=np.float32)
    np.testing.assert_allclose(editorial.unsharp_mask(constant), constant, atol=1e-7)


@pytest.mark.parametrize("target", [30.0, 210.0, 350.0])
@pytest.mark.parametrize("band", ["shadows", "highlights"])
def test_split_tone_is_rgb_continuous_across_opposing_hue(editorial, target, band):
    import colorsys

    # A tiny input change must not create the old opposing-hue color jump.
    opposite = (target + 180) % 360
    value = 0.2 if band == "shadows" else 0.9
    image = np.array(
        [[colorsys.hsv_to_rgb((opposite + offset) / 360, 0.3, value) for offset in (-0.001, 0.001)]],
        dtype=np.float32,
    )
    output = (
        editorial.split_tone(image, target, 0.06, None, 0)
        if band == "shadows"
        else editorial.split_tone(image, None, 0, target, 0.06)
    )
    assert np.max(np.abs(output[0, 0] - output[0, 1])) < 1e-4
    assert np.isfinite(output).all()


def test_split_tone_strength_tends_to_identity(editorial):
    image = np.array([[[0.2, 0.16, 0.12]]], dtype=np.float32)
    assert editorial.split_tone(image, 210, 0, 38, 0) is image
    output = editorial.split_tone(image, 210, 1e-7, None, 0)
    np.testing.assert_allclose(output, image, atol=1e-6, rtol=0)


@pytest.mark.parametrize("value", [0.0, 0.2, 0.8])
def test_split_tone_achromatic_input_has_defined_strength(editorial, value):
    import colorsys

    image = np.full((1, 1, 3), value, dtype=np.float32)
    alpha = 0.06 * max(1 - value * 2, 0)
    expected = colorsys.hsv_to_rgb(210 / 360, alpha, value)
    output = editorial.split_tone(image, 210, 0.06, None, 0)
    np.testing.assert_allclose(output[0, 0], expected, atol=1e-7, rtol=0)


def test_split_tone_opposing_chroma_cancellation_is_continuous(editorial):
    import colorsys

    image = np.array(
        [[colorsys.hsv_to_rgb((30 + offset) / 360, 1, 0.2) for offset in (-0.001, 0, 0.001)]],
        dtype=np.float32,
    )
    # At the center, the weighted opposing vectors cancel to gray. Recovering
    # both angle and magnitude keeps the resulting RGB continuous there.
    center_luma = np.sum(image[0, 1] * editorial.PROPHOTO_TO_XYZ[1])
    strength = 0.5 / (1 - 2 * center_luma)
    output = editorial.split_tone(image, 210, strength, None, 0)
    np.testing.assert_allclose(output[0, 1], [0.2, 0.2, 0.2], atol=1e-6, rtol=0)
    assert np.max(np.abs(output[0] - output[0, 1])) < 1e-4


def test_float_dust_reconstruction_does_not_ring_across_unit_range(editorial, monkeypatch):
    pytest.importorskip("cv2")
    image = np.full((64, 64, 3), 0.95, dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:35, 28:35] = 255
    # Isolate reconstruction with a known dust mask. OpenCV's real Telea path
    # must preserve a nearly constant field; unscaled unit floats clip to 0/1.
    monkeypatch.setattr(editorial.cv2, "threshold", lambda *_args: (220, mask.copy()))
    result = editorial.remove_dust_spots(image)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, image, atol=0.01, rtol=0)


def test_float_dust_removal_preserves_unmasked_sub_8bit_samples(editorial):
    pytest.importorskip("cv2")
    ramp = np.linspace(0.91, 0.96, 64 * 64, dtype=np.float32).reshape((64, 64))
    image = np.repeat(ramp[..., None], 3, axis=-1)
    image[28:35, 28:35] = 0
    result = editorial.remove_dust_spots(image)
    # Exercise real spot detection and reconstruction, including finite/bounded
    # samples and no introduced black/white clipping in the repaired center.
    assert np.isfinite(result).all()
    assert np.all((result >= 0) & (result <= 1))
    assert np.all((result[30:33, 30:33] > 0.2) & (result[30:33, 30:33] < 0.99))
    np.testing.assert_allclose(result[:20], image[:20], atol=1e-7, rtol=0)
    assert np.unique(result[:20, :, 0]).size > 256


def test_export_assets_color_and_format_contract(editorial, tmp_path):
    from PIL import Image

    image = np.full((32, 48, 3), 0.18, dtype=np.float32)
    print_path, web_path = tmp_path / "print.tif", tmp_path / "web.jpg"
    editorial.export_assets(image, print_path, web_path, None, None, 2500, 100, 0, 0)
    np.testing.assert_allclose(editorial.load_image_float(print_path), image, atol=1 / 65535)
    with Image.open(web_path) as web:
        assert web.format == "JPEG"
        assert web.info.get("icc_profile")
        np.testing.assert_allclose(np.asarray(web) / 255, 0.461356, atol=1 / 255)


def test_hdr_uses_measured_exposures_and_preserves_rgb(editorial, monkeypatch):
    paths = [Path("short.raw"), Path("long.raw")]
    radiance = np.array([[[0.2, 0.3, 0.4], [0.4, 0.2, 0.1]]], dtype=np.float32)
    monkeypatch.setattr(editorial, "_hdr_capture_settings", lambda path: ((0.5 if path == paths[0] else 1), 8.0, 100.0))
    monkeypatch.setattr(editorial, "raw_to_prophoto_tiff", lambda path: radiance * (0.5 if path == paths[0] else 1))
    result = editorial.hdr_merge_debvec(paths)
    reference = radiance * 0.75
    peak = reference.max(axis=-1, keepdims=True)
    expected = reference * (np.log1p(peak) / np.log(2)) / peak
    np.testing.assert_allclose(result, expected, atol=1e-7)
    # Exposure ordering does not change the result and RGB channels are not swapped.
    np.testing.assert_allclose(editorial.hdr_merge_debvec(paths[::-1]), result, atol=1e-7)
    assert result[0, 0, 2] > result[0, 0, 0]


def test_hdr_rejects_unmeasured_or_changed_capture_settings(editorial, monkeypatch, tmp_path):
    path = tmp_path / "missing.raw"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(editorial, "exifread", None)
    with pytest.raises(RuntimeError, match="verify shutter"):
        editorial._hdr_capture_settings(path)
    paths = [Path("a.raw"), Path("b.raw")]
    monkeypatch.setattr(editorial, "_hdr_capture_settings", lambda _: (1.0, 8.0, 100.0))
    with pytest.raises(ValueError, match="distinct"):
        editorial.hdr_merge_debvec(paths)
    monkeypatch.setattr(editorial, "_hdr_capture_settings", lambda p: ((1, 8, 100) if p == paths[0] else (2, 11, 100)))
    with pytest.raises(ValueError, match="aperture and ISO"):
        editorial.hdr_merge_debvec(paths)


def test_panorama_warps_float_originals_without_8bit_quantization(editorial, monkeypatch):
    pytest.importorskip("cv2")
    original = np.linspace(0.1, 0.8, 48 * 192 * 3, dtype=np.float32).reshape((48, 192, 3))
    paths = [Path("left.raw"), Path("right.raw")]
    monkeypatch.setattr(
        editorial, "raw_to_prophoto_tiff", lambda path: original[:, :128] if path == paths[0] else original[:, 64:]
    )
    monkeypatch.setattr(
        editorial, "_panorama_registration", lambda *_: np.array([[1, 0, 64], [0, 1, 0], [0, 0, 1]], dtype=float)
    )
    result = editorial.stitch_pano(paths)
    assert result.shape == original.shape
    np.testing.assert_allclose(result, original, atol=1e-7)
    assert np.unique(result[..., 0]).size > 256


def test_panorama_rejects_unbounded_geometry(editorial, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setattr(editorial, "raw_to_prophoto_tiff", lambda _: np.zeros((32, 64, 3), dtype=np.float32))
    monkeypatch.setattr(
        editorial, "_panorama_registration", lambda *_: np.array([[1, 0, 99999], [0, 1, 0], [0, 0, 1]], dtype=float)
    )
    with pytest.raises(ValueError, match="bounded warp"):
        editorial.stitch_pano([Path("a.raw"), Path("b.raw")])


def test_linear_contrast_preserves_shadows_neutrality_and_channel_ratios(editorial):
    ramp = np.linspace(0, 1, 2048, dtype=np.float32)[None, :, None]
    gray = np.repeat(ramp, 3, axis=-1)
    result = editorial.s_curve(editorial.adjust_contrast(gray, 6))
    assert np.all(np.diff(result[0, :, 0]) > 0)
    assert np.all(result[0, 1:-1] > 0)
    np.testing.assert_allclose(result[..., 0], result[..., 1], atol=1e-7)
    np.testing.assert_allclose(result[:, 0], 0, atol=1e-7)
    np.testing.assert_allclose(result[:, -1], 1, atol=1e-7)
    shadow = np.array([[[0.02, 0.03, 0.04]]], dtype=np.float32)
    adjusted = editorial.s_curve(editorial.adjust_contrast(shadow, 6))
    assert np.all(adjusted > 0)
    np.testing.assert_allclose(adjusted / adjusted[..., 1:2], shadow / shadow[..., 1:2], atol=1e-6)


@pytest.mark.parametrize("decode_failure", [False, True])
def test_editorial_cli_reports_real_completion_and_retains_manifest_paths(editorial, monkeypatch, tmp_path, decode_failure):
    import json

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "scene.DNG").write_bytes(b"unmodified source fixture")
    project = tmp_path / "project"
    config = {
        "project_name": "Fixture",
        "project_root": str(project),
        "input_raw_dir": str(input_dir),
        "rename": {"enabled": False},
        "selects": {"use_csv": False},
        "icc": {},
        "processing": {"workers": 1, "auto_upright": False},
        "styles": {"natural": {"exposure": 0, "contrast": 0, "saturation": 0}},
        "consistency": {"target_median": 0.42, "wb_neutralize": False},
        "export": {"web_long_edge_px": 100, "jpeg_quality": 96, "sharpen_web_amount": 0, "sharpen_print_amount": 0},
        "metadata": {},
        "deliver": {"zip": False},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(editorial.yaml.safe_dump(config))

    def decode(_path):
        if decode_failure:
            raise ValueError("fixture decode failure")
        return np.full((24, 32, 3), 0.2, dtype=np.float32)

    monkeypatch.setattr(editorial, "raw_to_prophoto_tiff", decode)
    # ReportLab is optional in the core-tier environment; its PDF output is
    # separately exercised by the native run, not faked as acceptance here.
    monkeypatch.setattr(editorial, "build_contact_sheet", lambda *args, **kwargs: None)
    assert editorial.main(["run", "--config", str(config_path)]) == (1 if decode_failure else 0)
    assert (input_dir / "scene.DNG").read_bytes() == b"unmodified source fixture"
    manifest = json.loads((project / "DOCS/Manifests/manifest.json").read_text())
    if decode_failure:
        assert manifest["exports"] == []
    else:
        assert manifest["exports"] == [
            {
                "style": "natural",
                "print_tif": "EXPORT/Print_TIFF/natural/scene.tif",
                "web_jpeg": "EXPORT/Web_JPEG/natural/scene.jpg",
            }
        ]
        assert (project / manifest["exports"][0]["print_tif"]).is_file()
        assert (project / manifest["exports"][0]["web_jpeg"]).is_file()


def test_exiftool_failure_is_not_available_or_successful_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_tool(monkeypatch)
    path = tmp_path / "export.jpg"
    module.save_jpeg_srgb(np.full((8, 8, 3), 0.4, dtype=np.float32), path, None)
    original = path.read_bytes()

    def fail(args, **kwargs):
        assert kwargs["check"] is True
        if args[-1] != "-ver":
            Path(args[-1]).write_bytes(b"interrupted metadata edit")
        raise module.subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(module.subprocess, "run", fail)
    assert module.has_exiftool() is False
    with pytest.raises(module.subprocess.CalledProcessError):
        module.embed_iptc_exiftool(path, {"creator": "Test photographer"})
    assert path.read_bytes() == original
    assert not path.with_suffix(".jpg.tmp").exists()


def test_metadata_without_optional_runtime_fails_explicitly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_tool(monkeypatch)
    monkeypatch.setattr(module, "piexif", None)
    with pytest.raises(RuntimeError, match="requires ExifTool"):
        module.embed_iptc_fallback_jpeg(tmp_path / "export.jpg", {"creator": "Photographer"})


def test_piexif_metadata_preserves_encoded_jpeg_and_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_tool(monkeypatch)
    piexif = pytest.importorskip("piexif", reason="Metadata insertion also runs in the governed editorial runtime")
    monkeypatch.setattr(module, "piexif", piexif)
    path = tmp_path / "export.jpg"
    samples = np.random.default_rng(9).random((16, 19, 3)).astype(np.float32)
    module.save_jpeg_srgb(samples, path, None)
    with module.Image.open(path) as image:
        before_samples = np.asarray(image).copy()
        before_profile = image.info["icc_profile"]
    # Add an existing unrelated EXIF field to verify merge rather than replacement.
    existing = piexif.dump({"0th": {piexif.ImageIFD.Software: b"Existing camera software"}})
    piexif.insert(existing, str(path))
    before_scan = path.read_bytes().split(b"\xff\xda", 1)[1]

    module.embed_iptc_fallback_jpeg(path, {"creator": "Photographer", "copyright": "Copyright test"})

    assert path.read_bytes().split(b"\xff\xda", 1)[1] == before_scan
    with module.Image.open(path) as image:
        np.testing.assert_array_equal(np.asarray(image), before_samples)
        assert image.info["icc_profile"] == before_profile
    tags = piexif.load(str(path))["0th"]
    assert tags[piexif.ImageIFD.Artist] == b"Photographer"
    assert tags[piexif.ImageIFD.Software] == b"Existing camera software"


def test_explicit_all_rejected_selects_does_not_process_every_raw(editorial, tmp_path):
    cfg = _cfg(editorial, tmp_path, selects={"use_csv": True})
    path = tmp_path / "DOCS" / "selects.csv"
    path.parent.mkdir()
    path.write_text("filename,keep,notes\nimage.DNG,0,\n")
    assert editorial.filter_selects(cfg, [Path("image.DNG")]) == []


@pytest.mark.parametrize("style_name", ["../../outside", "/absolute", "..", "", "nested\\style"])
def test_style_directory_cannot_escape_project(editorial, tmp_path, style_name):
    cfg = _cfg(editorial, tmp_path)
    cfg.styles = {style_name: {}}
    with pytest.raises(ValueError, match="single directory name"):
        cfg.validate()


def test_duplicate_raw_basename_rejected_before_offload(editorial, tmp_path):
    cfg = _cfg(editorial, tmp_path)
    for directory, content in (("first", b"first capture"), ("second", b"second capture")):
        raw = cfg.input_raw_dir / directory / "image.DNG"
        raw.parent.mkdir()
        raw.write_bytes(content)
    layout = editorial.Layout.build(cfg)
    with pytest.raises(ValueError, match="unique basenames"):
        editorial.mirror_offload(cfg, layout)
    assert not layout.RAW_ORIG.exists()


def test_raw_discovery_includes_full_vendor_suffixes(editorial, tmp_path):
    expected = []
    for name in ("nikon.NEF", "fuji.raf", "olympus.ORF", "canon.CR3"):
        path = tmp_path / name
        path.touch()
        expected.append(path)
    assert set(editorial.find_raws(tmp_path)) == set(expected)


def test_atomic_write_does_not_follow_predictable_temporary_symlink(editorial, tmp_path):
    destination = tmp_path / "export.tif"
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"preserve me")
    destination.with_suffix(".tif.tmp").symlink_to(unrelated)
    editorial.atomic_write(destination, lambda path: path.write_bytes(b"new output"))
    assert destination.read_bytes() == b"new output"
    assert unrelated.read_bytes() == b"preserve me"
    assert list(tmp_path.glob(".export.tif.*.tmp")) == []


@pytest.mark.parametrize("bad_header", ["pcs", "illuminant", "media_white"])
def test_profile_header_must_match_xyz_d50_contract(editorial, tmp_path, bad_header):
    import struct

    profile = bytearray(editorial.linear_prophoto_icc())
    if bad_header == "pcs":
        profile[20:24] = b"Lab "
    elif bad_header == "illuminant":
        profile[68:80] = struct.pack(">3i", 65536, 65536, 65536)
    else:
        for index in range(struct.unpack_from(">I", profile, 128)[0]):
            signature, offset, _size = struct.unpack_from(">4sII", profile, 132 + 12 * index)
            if signature == b"wtpt":
                profile[offset + 8 : offset + 20] = struct.pack(">3i", 65536, 65536, 65536)
    path = tmp_path / "master.tif"
    with pytest.raises(ValueError, match="connection|white point"):
        editorial.save_tiff16_prophoto(np.zeros((2, 2, 3)), path, bytes(profile))
    assert not path.exists()


def test_selected_raw_output_stem_collision_fails_before_decoding(editorial, monkeypatch, tmp_path):
    cfg = _cfg(editorial, tmp_path, selects={"use_csv": False})
    monkeypatch.setattr(editorial.PipelineConfig, "from_yaml", lambda _: cfg)
    monkeypatch.setattr(editorial, "mirror_offload", lambda *_: [Path("scene.DNG"), Path("scene.CR2")])
    with pytest.raises(ValueError, match="unique stems"):
        editorial.run_pipeline(tmp_path / "config.yml")


def test_contact_sheet_failure_preserves_prior_pdf_instead_of_silently_skipping(editorial, tmp_path):
    bad_image = tmp_path / "bad.jpg"
    bad_image.write_bytes(b"not an image")
    pdf = tmp_path / "contact.pdf"
    pdf.write_bytes(b"prior complete PDF")
    with pytest.raises(RuntimeError, match="failed to render bad.jpg"):
        editorial.build_contact_sheet([bad_image], pdf)
    assert pdf.read_bytes() == b"prior complete PDF"
    assert list(tmp_path.glob(".contact.pdf.*.tmp")) == []


def test_explicit_missing_metadata_csv_fails_configuration(editorial, tmp_path):
    cfg = _cfg(editorial, tmp_path, metadata={"csv_path": "DOCS/metadata.csv"})
    with pytest.raises(ValueError, match="Configured metadata CSV does not exist"):
        cfg.validate()


def _stub_image_lifetime_pipeline(editorial, monkeypatch, tmp_path, *, retouch, dtype):
    """Exercise real stage control flow with tiny arrays, never RAW decoding."""
    import weakref

    cfg = _cfg(editorial, tmp_path / "project", selects={"use_csv": False})
    cfg.processing["auto_upright"] = True
    cfg.styles = {"natural": {}, "minimal": {}}
    cfg.consistency["wb_neutralize"] = False
    cfg.retouch = {"dust_remove": retouch, "hotspot_reduce": False}
    raws = [Path("one.DNG"), Path("two.DNG")]
    references = {}
    reloads = {}
    normalized_writes = {}
    state = {"phase": None, "export_count": 0}

    def track(kind, image):
        references.setdefault(kind, []).append(weakref.ref(image))
        return image

    def released(*kinds):
        for kind in kinds:
            assert all(ref() is None for ref in references.get(kind, [])), kind

    def samples(path):
        return np.full((3, 4, 3), 0.1 if Path(path).stem == "one" else 0.7, dtype=dtype)

    def decode(path):
        released("decode")
        return track("decode", samples(path))

    def load(path):
        path = Path(path)
        released("decode", "upright_input", "upright_output")
        if path.parent.name == "BaseTIFF":
            phase = "upright_input"
        elif path.parent.name == "Aligned":
            released("base", "graded")
            phase = "base"
        else:
            reloads[path] = reloads.get(path, 0) + 1
            if reloads[path] == 1:
                released("base", "graded", "normalization_input", "normalization_output")
                phase = "normalization_input"
            elif retouch and reloads[path] == 2:
                released("normalization_input", "normalization_output", "retouch_input", "retouch_output")
                phase = "retouch_input"
            else:
                released("normalization_input", "normalization_output", "retouch_input", "retouch_output", "export_input")
                phase = "export_input"
        state["phase"] = phase
        return track(phase, samples(path))

    def upright(image, *_args):
        return track("upright_output", image.copy())

    def grade(image, *_args):
        released("graded")
        state["phase"] = "graded"
        return track("graded", image.copy())

    original_normalize = editorial.normalize_exposure

    def normalize(images, **kwargs):
        assert len(images) == 1, "Normalization must decode/process one image at a time"
        state["phase"] = "normalization_output"
        result = original_normalize(images, **kwargs)
        track("normalization_output", result[0])
        return result

    def save(image, path, _profile):
        # The real writer creates destinations that the normalization preflight
        # stats; tiny marker bytes supply only that filesystem identity here.
        Path(path).write_bytes(b"synthetic encoded TIFF stand-in")
        if state["phase"] == "normalization_output":
            normalized_writes[Path(path)] = image.copy()

    def dust(image):
        state["phase"] = "retouch_output"
        return track("retouch_output", image.copy())

    def export(_image, *_args):
        released("decode", "base", "graded", "normalization_input", "normalization_output", "retouch_input", "retouch_output")
        state["export_count"] += 1

    def contact_sheet(*_args, **_kwargs):
        released(*references)

    monkeypatch.setattr(editorial.PipelineConfig, "from_yaml", lambda _: cfg)
    monkeypatch.setattr(editorial, "mirror_offload", lambda *_: raws)
    monkeypatch.setattr(editorial, "rename_raws", lambda *_: raws)
    monkeypatch.setattr(editorial, "ensure_selects_csv", lambda *_: tmp_path / "unused-selects.csv")
    monkeypatch.setattr(editorial, "filter_selects", lambda *_: raws)
    monkeypatch.setattr(editorial, "tqdm", lambda items, **_kwargs: items)
    monkeypatch.setattr(editorial, "raw_to_prophoto_tiff", decode)
    monkeypatch.setattr(editorial, "load_image_float", load)
    monkeypatch.setattr(editorial, "auto_upright_small", upright)
    monkeypatch.setattr(editorial, "style_grade", grade)
    monkeypatch.setattr(editorial, "normalize_exposure", normalize)
    monkeypatch.setattr(editorial, "save_tiff16_prophoto", save)
    monkeypatch.setattr(editorial, "remove_dust_spots", dust)
    monkeypatch.setattr(editorial, "export_assets", export)
    monkeypatch.setattr(editorial, "build_contact_sheet", contact_sheet)
    return cfg, raws, normalized_writes, state, original_normalize


@pytest.mark.parametrize("retouch", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pipeline_releases_stage_arrays_and_preserves_streamed_normalization(editorial, monkeypatch, tmp_path, retouch, dtype):
    cfg, raws, actual, state, normalize = _stub_image_lifetime_pipeline(
        editorial, monkeypatch, tmp_path, retouch=retouch, dtype=dtype
    )
    # Original batch formula is independent per image. Preserve exact dtype and
    # values while eliminating the simultaneously retained image/result lists.
    expected = normalize(
        [np.full((3, 4, 3), value, dtype=dtype) for value in (0.1, 0.7)],
        target_median=cfg.consistency["target_median"],
    )

    editorial.run_pipeline(tmp_path / "unused-config.yml")

    assert state["export_count"] == 4
    assert len(actual) == 4
    for style in cfg.styles:
        for raw, reference in zip(raws, expected):
            saved = actual[cfg.project_root / "WORK" / "Variants" / style / raw.with_suffix(".tif").name]
            assert saved.dtype == reference.dtype
            np.testing.assert_array_equal(saved, reference)


def test_streamed_normalization_failure_does_not_publish_success(editorial, monkeypatch, tmp_path):
    cfg, _raws, actual, state, _normalize = _stub_image_lifetime_pipeline(
        editorial, monkeypatch, tmp_path, retouch=False, dtype=np.float32
    )
    original_load = editorial.load_image_float

    def fail_later_input(path):
        path = Path(path)
        if path.parent.name == "natural" and path.name == "two.tif":
            raise ValueError("synthetic later normalization input failure")
        return original_load(path)

    monkeypatch.setattr(editorial, "load_image_float", fail_later_input)
    with pytest.raises(ValueError, match="later normalization input failure"):
        editorial.run_pipeline(tmp_path / "unused-config.yml")

    # A streamed phase may retain a completed prefix, but may not present it as
    # a complete export or successful final manifest.
    assert len(actual) == 1
    assert state["export_count"] == 0
    assert not (cfg.project_root / "DOCS" / "Manifests" / "manifest.json").exists()


@pytest.mark.parametrize(
    "duplicate_source,fail_at",
    [
        ("repeated_pano_group", None),
        ("generated_stem_collision", None),
        ("hardlink_destinations", None),
        ("case_alias_destinations", None),
        ("repeated_pano_group", 2),
    ],
)
def test_streamed_normalization_preserves_duplicate_destination_bytes(
    editorial, monkeypatch, tmp_path, duplicate_source, fail_at
):
    if duplicate_source == "case_alias_destinations":
        probe = tmp_path / "case-probe"
        probe.write_bytes(b"case sensitivity fixture")
        if not (tmp_path / "CASE-PROBE").exists():
            pytest.skip("This filesystem does not support distinct-case aliases of one destination")

    cfg = _cfg(editorial, tmp_path / "project", selects={"use_csv": False})
    raws = [Path("one.DNG"), Path("two.DNG")]
    groups = [["one.DNG", "two.DNG"]]
    names = ["one.tif", "two.tif", "one_PANO.tif", "one_PANO.tif"]
    if duplicate_source == "repeated_pano_group":
        groups.append(groups[0].copy())
    elif duplicate_source == "generated_stem_collision":
        # Unique RAW stems may still collide with a generated pano filename.
        raws.append(Path("one_PANO.DNG"))
    elif duplicate_source == "case_alias_destinations":
        raws.append(Path("ONE_pano.DNG"))
        names = ["one.tif", "two.tif", "ONE_pano.tif", "one_PANO.tif"]
    else:
        groups = []
        names = ["one.tif", "two.tif"]
    cfg.processing.update({"auto_upright": False, "enable_pano": bool(groups), "pano_groups": groups})
    cfg.consistency["wb_neutralize"] = False
    image = np.full((3, 4, 3), 0.1, dtype=np.float32)
    original_normalize = editorial.normalize_exposure
    original_save = editorial.save_tiff16_prophoto
    original_copy = editorial.shutil.copyfile
    original_temporary_directory = editorial.tempfile.TemporaryDirectory
    exports = []
    normalization_calls = []
    normalization_writes = []
    snapshots = []
    scratch_directories = []

    # Reference the previous batch read-before-write behavior, including uint16
    # TIFF quantization. Every duplicate begins with the same original pixels.
    reference_input = tmp_path / "reference-input.tif"
    editorial.save_tiff16_prophoto(image, reference_input, None)
    source = editorial.load_image_float(reference_input)
    factor = np.clip(cfg.consistency["target_median"] / (editorial.median_luma(source) + 1e-6), 0.6, 1.6)
    assert factor == 1.6, "The fixture must expose repeated clamped normalization"
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    batch_outputs = original_normalize([source.copy() for _ in names], target_median=cfg.consistency["target_median"])
    for name, normalized in zip(names, batch_outputs):
        editorial.save_tiff16_prophoto(normalized, reference_dir / name, None)

    # A second read/normalize/write is observably different, so a duplicate-path
    # regression cannot pass by coincidence or only by counting calls.
    twice = original_normalize(
        [editorial.load_image_float(reference_dir / names[-1])],
        target_median=cfg.consistency["target_median"],
    )[0]
    twice_path = tmp_path / "twice.tif"
    editorial.save_tiff16_prophoto(twice, twice_path, None)
    assert twice_path.read_bytes() != (reference_dir / names[-1]).read_bytes()

    def normalize(images, **kwargs):
        assert len(images) == 1
        normalization_calls.append(1)
        if fail_at == len(normalization_calls):
            raise ValueError("synthetic failure after shared-identity snapshot")
        return original_normalize(images, **kwargs)

    def save(samples, path, profile):
        path = Path(path)
        original_save(samples, path, profile)
        if normalization_calls:
            normalization_writes.append(path.name)
        elif duplicate_source == "hardlink_destinations" and path.parent.name == "natural" and path.name == "two.tif":
            path.unlink()
            path.hardlink_to(path.with_name("one.tif"))

    def snapshot(source_path, frozen_path, *_args, **_kwargs):
        assert not normalization_calls, "All shared identities must be frozen before the first normalization write"
        snapshots.append(Path(source_path))
        return original_copy(source_path, frozen_path)

    def temporary_directory(*args, **kwargs):
        # Keep the synthetic fixture's scratch under its caller-owned tmp root.
        directory = original_temporary_directory(*args, dir=tmp_path, **kwargs)
        scratch_directories.append(Path(directory.name))
        return directory

    def export(_image, print_path, _web_path, *_args):
        exports.append(Path(print_path).name)

    monkeypatch.setattr(editorial.PipelineConfig, "from_yaml", lambda _: cfg)
    monkeypatch.setattr(editorial, "mirror_offload", lambda *_: raws)
    monkeypatch.setattr(editorial, "rename_raws", lambda *_: raws)
    monkeypatch.setattr(editorial, "ensure_selects_csv", lambda *_: tmp_path / "unused-selects.csv")
    monkeypatch.setattr(editorial, "filter_selects", lambda *_: raws)
    monkeypatch.setattr(editorial, "tqdm", lambda items, **_kwargs: items)
    monkeypatch.setattr(editorial, "raw_to_prophoto_tiff", lambda _: image.copy())
    monkeypatch.setattr(editorial, "stitch_pano", lambda _: image.copy())
    monkeypatch.setattr(editorial, "style_grade", lambda samples, *_: samples.copy())
    monkeypatch.setattr(editorial, "normalize_exposure", normalize)
    monkeypatch.setattr(editorial, "save_tiff16_prophoto", save)
    monkeypatch.setattr(editorial.shutil, "copyfile", snapshot)
    monkeypatch.setattr(editorial.tempfile, "TemporaryDirectory", temporary_directory)
    monkeypatch.setattr(editorial, "export_assets", export)
    monkeypatch.setattr(editorial, "build_contact_sheet", lambda *_args, **_kwargs: None)

    if fail_at:
        with pytest.raises(ValueError, match="after shared-identity snapshot"):
            editorial.run_pipeline(tmp_path / "unused-config.yml")
        assert len(snapshots) == 1
        assert all(not path.exists() for path in scratch_directories)
        assert normalization_writes == names[: fail_at - 1]
        assert exports == []
        assert not (cfg.project_root / "DOCS" / "Manifests" / "manifest.json").exists()
        return

    editorial.run_pipeline(tmp_path / "unused-config.yml")

    actual_dir = cfg.project_root / "WORK" / "Variants" / "natural"
    assert len(snapshots) == 1, "Copy one encoded input only for the shared identity, never every input"
    assert all(not path.exists() for path in scratch_directories)
    assert len(normalization_calls) == len(names)
    assert normalization_writes == names, "Every original destination must still be written in its original order"
    assert exports == names, "Normalization must not deduplicate or reorder exports"
    for name in set(names):
        assert (actual_dir / name).read_bytes() == (reference_dir / name).read_bytes()
    manifest_path = cfg.project_root / "DOCS" / "Manifests" / "manifest.json"
    import json

    assert [Path(entry["print_tif"]).name for entry in json.loads(manifest_path.read_text())["exports"]] == names


@pytest.mark.parametrize("print_amount", [0, 0.1])
def test_export_drops_separate_print_buffer_before_web(editorial, monkeypatch, tmp_path, print_amount):
    import weakref

    image = np.full((3, 4, 3), 0.2, dtype=np.float32)
    print_reference = weakref.ref(image)
    calls = []

    def sharpen(samples, *, amount, **_kwargs):
        nonlocal print_reference
        calls.append(amount)
        result = samples if amount == 0 else samples.copy()
        if len(calls) == 1:
            print_reference = weakref.ref(result)
        return result

    def resize(samples, edge):
        assert samples is image and edge == 2500
        if print_amount:
            assert print_reference() is None, "Print-only buffer survives into web construction"
        else:
            assert print_reference() is image, "Zero sharpening must preserve the caller-owned alias"
        return samples.copy()

    monkeypatch.setattr(editorial, "unsharp_mask", sharpen)
    monkeypatch.setattr(editorial, "save_tiff16_prophoto", lambda *_: None)
    monkeypatch.setattr(editorial, "resize_long_edge", resize)
    monkeypatch.setattr(editorial, "save_jpeg_srgb", lambda *_args, **_kwargs: None)

    editorial.export_assets(image, tmp_path / "print.tif", tmp_path / "web.jpg", None, None, 2500, 96, 0.35, print_amount)

    assert calls == [print_amount, 0.35]
    np.testing.assert_array_equal(image, np.full((3, 4, 3), 0.2, dtype=np.float32))


@pytest.mark.parametrize("gamma_profile", [False, True])
def test_export_lifetime_change_preserves_synthetic_encoded_bytes(editorial, tmp_path, gamma_profile):
    image = np.random.default_rng(11).random((12, 17, 3)).astype(np.float32)
    profile = _gamma_prophoto_profile(editorial) if gamma_profile else None
    reference_print, reference_web = tmp_path / "reference.tif", tmp_path / "reference.jpg"
    actual_print, actual_web = tmp_path / "actual.tif", tmp_path / "actual.jpg"

    # Previous export sequence, including its longer-lived print-only buffer.
    # The change may release that buffer, but must preserve all export operations.
    print_image = editorial.unsharp_mask(image, amount=0.1, radius=1.2, threshold=0.0)
    editorial.save_tiff16_prophoto(print_image, reference_print, profile)
    web_image = editorial.resize_long_edge(image, 10)
    web_image = editorial.unsharp_mask(web_image, amount=0.35, radius=1.2, threshold=0.0)
    editorial.save_jpeg_srgb(editorial.linear_prophoto_to_srgb(web_image), reference_web, None, quality=96)

    editorial.export_assets(image, actual_print, actual_web, profile, None, 10, 96, 0.35, 0.1)

    assert actual_print.read_bytes() == reference_print.read_bytes()
    assert actual_web.read_bytes() == reference_web.read_bytes()
