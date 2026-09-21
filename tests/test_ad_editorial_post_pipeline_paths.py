from __future__ import annotations

import importlib.util
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
