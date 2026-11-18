from __future__ import annotations

import errno
import hashlib
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import pytest

pytestmark = pytest.mark.unit

# pylint: disable=no-member  # False positives with package imports


# NOTE: Tests assume package is installed with: pip install -e .

try:
    from .documentation import documents
except ImportError:  # pragma: no cover - fallback for direct execution
    from tests.documentation import documents

np = pytest.importorskip("numpy")
pytest.importorskip("PIL.Image")
pytest.importorskip("PIL.TiffImagePlugin")
from PIL import Image, TiffImagePlugin  # noqa: E402  # pylint: disable=wrong-import-position

import luxury_tiff_batch_processor as ltiff  # noqa: E402  # pylint: disable=wrong-import-position,consider-using-from-import
from luxury_tiff_batch_processor import cli as ltiff_cli  # noqa: E402  # pylint: disable=wrong-import-position
from luxury_tiff_batch_processor import io_utils  # noqa: E402  # pylint: disable=wrong-import-position
from luxury_tiff_batch_processor import pipeline  # noqa: E402  # pylint: disable=wrong-import-position


def test_run_pipeline_exposed_in_dunder_all():
    exported = getattr(ltiff, "__all__", ())
    assert "run_pipeline" in exported


def test_cli_module_importable():
    """Test that the CLI module can be imported."""
    assert hasattr(ltiff_cli, "main"), "CLI module should have a main function"


def test_cli_help_works():
    """Test that the CLI help command works."""
    result = subprocess.run(
        [sys.executable, "-m", "luxury_tiff_batch_processor.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Batch enhance TIFF files" in result.stdout or "Usage" in result.stdout


def _saturation(rgb: np.ndarray) -> np.ndarray:
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    return maxc - minc


@documents("Material Response honors surface physics, not global transforms")
def test_apply_adjustments_respects_exposure_clamp():
    arr = np.full((4, 4, 3), 0.25, dtype=np.float32)
    settings = ltiff.AdjustmentSettings(exposure=2.0)

    out = ltiff.apply_adjustments(arr, settings)

    assert out.shape == arr.shape
    assert np.allclose(out, 1.0, atol=1e-4)


@documents("Token system allows composition without prescription")
def test_apply_adjustments_vibrance_boosts_muted_colors_more():
    neutral = np.full((1, 1, 3), 0.5, dtype=np.float32)
    muted = np.array([[[0.5, 0.35, 0.3]]], dtype=np.float32)
    saturated = np.array([[[0.9, 0.2, 0.2]]], dtype=np.float32)
    arr = np.concatenate([neutral, muted, saturated], axis=0)

    settings = ltiff.AdjustmentSettings(vibrance=0.8)
    out = ltiff.apply_adjustments(arr, settings)

    sat_before = _saturation(arr)
    sat_after = _saturation(out)

    # Neutral remains essentially neutral
    assert sat_after[0, 0] == pytest.approx(sat_before[0, 0], abs=1e-4)
    # Muted pixel receives a larger relative boost than already saturated pixel
    muted_gain = sat_after[1, 0] - sat_before[1, 0]
    saturated_gain = sat_after[2, 0] - sat_before[2, 0]
    assert muted_gain > saturated_gain


def test_processing_profiles_balance_costly_filters():
    rng = np.random.default_rng(2024)
    arr = rng.random((12, 12, 3), dtype=np.float32)
    adjustments = ltiff.AdjustmentSettings(chroma_denoise=0.7, glow=0.5)

    quality = ltiff.PROCESSING_PROFILES["quality"]
    balanced = ltiff.PROCESSING_PROFILES["balanced"]
    performance = ltiff.PROCESSING_PROFILES["performance"]

    quality_result = ltiff.apply_adjustments(arr, adjustments, profile=quality)
    balanced_result = ltiff.apply_adjustments(arr, adjustments, profile=balanced)
    performance_result = ltiff.apply_adjustments(arr, adjustments, profile=performance)

    assert not np.allclose(quality_result, arr)
    assert not np.allclose(balanced_result, arr)
    assert np.allclose(performance_result, arr, atol=1e-6)

    quality_delta = float(np.mean(np.abs(quality_result - arr)))
    balanced_delta = float(np.mean(np.abs(balanced_result - arr)))
    assert balanced_delta < quality_delta
    assert balanced_delta > 0.0


@documents("Pipeline preserves authored intent while optimizing logistics")
def test_process_single_image_handles_resize_and_metadata(tmp_path: Path):
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()

    arr = np.linspace(0, 255, 4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    # Suppress deprecated mode parameter warning for cleaner test output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        image = Image.fromarray(arr, mode="RGB")
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[270] = "Luxury scene"
    source_path = source_dir / "frame.tif"
    image.save(source_path, tiffinfo=info)

    dest_path = output_dir / "frame_processed.tif"

    ltiff.process_single_image(
        source_path,
        dest_path,
        ltiff.AdjustmentSettings(),
        compression="tiff_lzw",
        resize_long_edge=2,
    )

    assert dest_path.exists()
    with Image.open(dest_path) as processed:
        assert processed.size == (2, 2)
        tags = getattr(processed, "tag_v2", {})
        assert tags.get(270) == "Luxury scene"
        assert np.array(processed).dtype == np.uint8


def test_process_single_image_profile_overrides_dtype_and_compression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()

    image = Image.new("RGB", (4, 4), color=(32, 64, 96))
    source_path = source_dir / "frame.tif"
    image.save(source_path)

    original_float_to_dtype = pipeline.float_to_dtype_array
    captured_dtypes: list[np.dtype] = []

    def spy_float_to_dtype_array(*args, **kwargs):
        dtype = np.dtype(args[1])
        captured_dtypes.append(dtype)
        return original_float_to_dtype(*args, **kwargs)

    monkeypatch.setattr(pipeline, "float_to_dtype_array", spy_float_to_dtype_array)

    original_save_image = pipeline.save_image
    captured_compressions: list[str] = []

    def spy_save_image(destination, arr_int, dtype, metadata, icc_profile, compression):
        captured_compressions.append(compression)
        return original_save_image(destination, arr_int, dtype, metadata, icc_profile, compression)

    monkeypatch.setattr(pipeline, "save_image", spy_save_image)

    for profile_name in ("quality", "balanced", "performance"):
        profile = ltiff.PROCESSING_PROFILES[profile_name]
        dest_path = output_dir / f"{profile_name}.tif"
        ltiff.process_single_image(
            source_path,
            dest_path,
            ltiff.AdjustmentSettings(),
            compression=profile.resolve_compression("tiff_lzw"),
            profile=profile,
        )

    assert [dtype.type for dtype in captured_dtypes] == [np.uint8, np.uint16, np.uint8]
    assert captured_compressions == ["tiff_lzw", "tiff_lzw", "tiff_jpeg"]


def test_process_single_image_cleanup_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()

    arr = np.linspace(0, 255, 4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        image = Image.fromarray(arr, mode="RGB")
    source_path = source_dir / "frame.tif"
    image.save(source_path)

    dest_path = output_dir / "frame_processed.tif"
    original_bytes = b"original-destination"
    dest_path.write_bytes(original_bytes)

    class Boom(RuntimeError):
        pass

    def failing_save(path: Path, *args, **kwargs) -> None:
        Path(path).write_bytes(b"partial")
        raise Boom("simulated failure")

    monkeypatch.setattr(pipeline, "save_image", failing_save)
    monkeypatch.setattr(ltiff, "save_image", failing_save)

    with pytest.raises(Boom):
        ltiff.process_single_image(
            source_path,
            dest_path,
            ltiff.AdjustmentSettings(),
            compression="tiff_lzw",
        )

    assert dest_path.exists()
    assert dest_path.read_bytes() == original_bytes
    temp_artifacts = list(dest_path.parent.glob(f".{dest_path.name}.tmp*"))
    assert not temp_artifacts


def test_save_image_round_trip_la_mode_pillow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(io_utils, "tifffile", None)

    luminance = np.arange(16, dtype=np.uint8).reshape((4, 4))
    alpha = np.flipud(luminance)
    arr = np.stack([luminance, alpha], axis=2)

    destination = tmp_path / "la.tif"
    ltiff.save_image(destination, arr, arr.dtype, metadata=None, icc_profile=None, compression="tiff_lzw")

    with Image.open(destination) as image:
        assert image.mode == "LA"
        restored = np.array(image)

    assert restored.shape == arr.shape
    assert np.array_equal(restored, arr)


def _fine_grained_uint16(shape: tuple[int, ...]) -> np.ndarray:
    """16-bit samples whose low bytes vary, so any 8-bit round trip is detectable."""

    rng = np.random.default_rng(1608)
    return rng.integers(0, 65536, size=shape, dtype=np.uint16)


@pytest.mark.parametrize("profile_name", ["quality", "balanced"])
@pytest.mark.parametrize("layout", ["contig", "planar", "rgba"])
def test_process_single_image_keeps_16bit_tiff_precision(tmp_path: Path, layout: str, profile_name: str):
    """H-08: Pillow decodes 16-bit RGB(A) TIFFs at 8 bits; the pipeline must not inherit that loss."""

    tifffile = pytest.importorskip("tifffile")
    channels = 4 if layout == "rgba" else 3
    samples = _fine_grained_uint16((10, 14, channels))
    source = tmp_path / f"{layout}16.tif"
    if layout == "planar":
        tifffile.imwrite(source, np.moveaxis(samples, -1, 0), photometric="rgb", planarconfig="separate")
    elif layout == "rgba":
        tifffile.imwrite(source, samples, photometric="rgb", extrasamples="unassalpha")
    else:
        tifffile.imwrite(source, samples, photometric="rgb")
    with Image.open(source) as image:  # the precondition the pipeline has to work around
        assert np.array(image).dtype == np.uint8

    destination = tmp_path / "out" / f"{layout}_{profile_name}.tif"
    ltiff.process_single_image(
        source,
        destination,
        ltiff.AdjustmentSettings(),
        compression="tiff_lzw",
        profile=ltiff.PROCESSING_PROFILES[profile_name],
    )

    with tifffile.TiffFile(destination) as written:
        page = written.pages[0]
        output = page.asarray()
        extrasamples = tuple(int(value) for value in page.extrasamples)
    assert output.dtype == np.uint16
    np.testing.assert_array_equal(output, samples)
    assert extrasamples == ((2,) if layout == "rgba" else ())


@pytest.mark.parametrize("profile_name", ["quality", "balanced", "performance"])
def test_process_single_image_8bit_output_is_unchanged_pillow_route(tmp_path: Path, profile_name: str):
    """8-bit sources keep the Pillow decode, so their deliverables stay byte-identical."""

    ImageCms = pytest.importorskip("PIL.ImageCms")  # pylint: disable=invalid-name

    arr = np.random.default_rng(8).integers(0, 256, size=(12, 18, 3), dtype=np.uint8)
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[270] = "Luxury scene"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    source = tmp_path / "frame8.tif"
    Image.fromarray(arr).save(source, compression="tiff_lzw", dpi=(300, 300), icc_profile=icc, tiffinfo=info)
    profile = ltiff.PROCESSING_PROFILES[profile_name]
    adjustments = ltiff.AdjustmentSettings(exposure=0.2, vibrance=0.3, clarity=0.2, glow=0.1)
    compression = profile.resolve_compression("tiff_lzw")

    destination = tmp_path / "out" / "frame8.tif"
    ltiff.process_single_image(source, destination, adjustments, compression=compression, profile=profile)

    # Reference: the Pillow-only decode route every 8-bit source took before H-08.
    reference = tmp_path / "reference.tif"
    with Image.open(source) as image:
        assert io_utils.read_full_precision_pixels(source, image) is None
        decoded = io_utils.image_to_float(image, return_format="object")
        adjusted = ltiff.apply_adjustments(decoded.array, adjustments, profile=profile)
        target_dtype = profile.target_dtype(decoded.dtype)
        arr_int = io_utils.float_to_dtype_array(adjusted, target_dtype, decoded.alpha, decoded.base_channels)
        io_utils.save_image(reference, arr_int, target_dtype, image.tag_v2, image.info.get("icc_profile"), compression)

    assert destination.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize(
    ("dpi", "unit", "expected_unit"),
    [((300, 300), None, 2), ((72.5, 144.25), None, 2), ((118, 59), "cm", 3)],
)
def test_process_single_image_16bit_output_keeps_dpi_and_single_icc(
    tmp_path: Path, dpi: tuple[float, float], unit: str | None, expected_unit: int
):
    """M-10: the tifffile (16-bit) writer kept neither DPI nor a single ICC tag."""

    tifffile = pytest.importorskip("tifffile")
    ImageCms = pytest.importorskip("PIL.ImageCms")  # pylint: disable=invalid-name

    arr = np.random.default_rng(10).integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    source = tmp_path / "frame8.tif"
    save_kwargs: Dict[str, Any] = {"compression": "tiff_lzw", "icc_profile": icc, "description": "Luxury"}
    if unit is None:
        save_kwargs["dpi"] = dpi
    else:
        save_kwargs.update(x_resolution=dpi[0], y_resolution=dpi[1], resolution_unit=unit)
    Image.fromarray(arr).save(source, **save_kwargs)
    with Image.open(source) as image:
        source_resolution = (image.tag_v2[282], image.tag_v2[283])
        assert int(image.tag_v2[296]) == expected_unit
        source_dpi = image.info["dpi"]

    destination = tmp_path / "out" / "frame16.tif"
    profile = ltiff.PROCESSING_PROFILES["balanced"]  # 8-bit sources become 16-bit masters
    ltiff.process_single_image(source, destination, ltiff.AdjustmentSettings(), compression="none", profile=profile)

    with tifffile.TiffFile(destination) as written:
        page = written.pages[0]
        assert page.dtype == np.uint16
        codes = [tag.code for tag in page.tags]
        resolution = (page.tags[282].value, page.tags[283].value)
        written_unit = int(page.tags[296].value)
        icc_values = [tag.value for tag in page.tags if tag.code == 34675]
        description = page.tags[270].value
    assert len(codes) == len(set(codes))
    assert icc_values == [icc]
    assert description == "Luxury"
    assert written_unit == expected_unit
    for written_value, source_value in zip(resolution, source_resolution):
        assert written_value[0] / written_value[1] == pytest.approx(float(source_value))
    with Image.open(destination) as reread:
        assert reread.info["dpi"] == pytest.approx(source_dpi)
        assert reread.info["icc_profile"] == icc


def test_save_image_16bit_writes_icc_from_metadata_when_no_profile_given(tmp_path: Path):
    """The tifffile writer keeps a metadata-only ICC tag once, as the Pillow writer does."""

    tifffile = pytest.importorskip("tifffile")
    icc = b"\x00\x01icc-profile-bytes"
    destination = tmp_path / "meta_icc.tif"
    io_utils.save_image(destination, _fine_grained_uint16((4, 6, 3)), np.uint16, {34675: icc, 270: "x"}, None, "none")
    with tifffile.TiffFile(destination) as written:
        page = written.pages[0]
        icc_values = [tag.value for tag in page.tags if tag.code == 34675]
        resolution_unit = int(page.tags[296].value)
    assert icc_values == [icc]
    assert resolution_unit == 1  # no source resolution: tifffile's own default stays


def test_save_image_8bit_output_matches_direct_pillow_save(tmp_path: Path):
    """M-10 fix is confined to the tifffile writer: 8-bit deliverables are byte-identical to Pillow's."""

    ImageCms = pytest.importorskip("PIL.ImageCms")  # pylint: disable=invalid-name
    arr = np.random.default_rng(11).integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    source = tmp_path / "src8.tif"
    Image.fromarray(arr).save(source, compression="tiff_lzw", dpi=(300, 300), icc_profile=icc, description="Luxury")
    with Image.open(source) as image:
        metadata = image.tag_v2
        destination = tmp_path / "via_save_image.tif"
        io_utils.save_image(destination, arr, np.uint8, metadata, icc, "tiff_lzw")
        reference = tmp_path / "direct_pillow.tif"
        safe = io_utils.sanitize_tiff_metadata(metadata)
        Image.fromarray(arr, mode="RGB").save(reference, format="TIFF", compression="tiff_lzw", tiffinfo=safe, icc_profile=icc)
    assert destination.read_bytes() == reference.read_bytes()
    with Image.open(destination) as reread:
        assert reread.info["dpi"] == pytest.approx((300.0, 300.0))


def test_read_full_precision_pixels_warns_without_tifffile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Without tifffile a 16-bit RGB source is still processed at 8 bits, but never silently."""

    tifffile = pytest.importorskip("tifffile")
    source = tmp_path / "rgb16.tif"
    tifffile.imwrite(source, _fine_grained_uint16((6, 8, 3)), photometric="rgb")
    monkeypatch.setattr(io_utils, "tifffile", None)

    with Image.open(source) as image, caplog.at_level("WARNING", logger="luxury_tiff_batch_processor"):
        assert io_utils.read_full_precision_pixels(source, image) is None

    assert "stores 16 bits per sample but is decoded at 8 bits" in caplog.text


@documents("Dry runs provide planning insight without side effects")
def test_run_pipeline_dry_run_creates_no_outputs(tmp_path: Path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    image = Image.new("RGB", (2, 2), color=(128, 128, 128))
    source_path = input_dir / "sample.tif"
    image.save(source_path)

    args = ltiff.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--dry-run",
        ]
    )

    processed = ltiff.run_pipeline(args)

    assert processed == 0
    assert not any(output_dir.rglob("*.tif"))


def _create_sample_image(path: Path) -> None:
    image = Image.new("RGB", (2, 2), color=(32, 64, 96))
    image.save(path)


def _collect_output_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()  # nosec B324 - test-only deterministic hash comparison
        for path in sorted(output_dir.glob("*.tif"))
    }


def test_run_pipeline_parallel_execution(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    for index in range(3):
        _create_sample_image(input_dir / f"frame_{index}.tif")

    args = ltiff.parse_args(
        [
            str(input_dir),
            str(output_dir),
            "--workers",
            "2",
        ]
    )

    processed = ltiff.run_pipeline(args)

    outputs = sorted(p.name for p in output_dir.glob("*.tif"))
    assert processed == 3
    assert outputs == [f"frame_{i}_lux.tif" for i in range(3)]


def test_run_pipeline_parallel_permissionerror_and_oserror_fallback_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(3):
        _create_sample_image(input_dir / f"frame_{index}.tif")

    class _ImmediateFuture:
        def __init__(self, value: bool) -> None:
            self._value = value

        def result(self) -> bool:
            return self._value

    class _InlineExecutor:
        def __init__(self, max_workers: int):  # pylint: disable=unused-argument
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001, ANN201
            return False

        def submit(self, fn, *args, **kwargs) -> _ImmediateFuture:  # noqa: ANN001, ANN002
            return _ImmediateFuture(fn(*args, **kwargs))

    def _as_completed_inline(futures):  # noqa: ANN001, ANN202
        yield from futures

    serial_output = tmp_path / "serial_output"
    serial_args = ltiff.parse_args([str(input_dir), str(serial_output), "--workers", "1", "--no-progress"])
    serial_processed = ltiff.run_pipeline(serial_args)
    serial_hashes = _collect_output_hashes(serial_output)

    monkeypatch.setattr(ltiff_cli, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(ltiff_cli, "as_completed", _as_completed_inline)
    parallel_output = tmp_path / "parallel_output"
    parallel_args = ltiff.parse_args([str(input_dir), str(parallel_output), "--workers", "2", "--no-progress"])
    parallel_processed = ltiff.run_pipeline(parallel_args)
    parallel_hashes = _collect_output_hashes(parallel_output)

    class _DeniedExecutor:
        def __init__(self, max_workers: int):  # pylint: disable=unused-argument
            raise PermissionError("simulated semaphore restriction")

    class _DeniedOSErrorExecutor:
        def __init__(self, max_workers: int):  # pylint: disable=unused-argument
            raise OSError(errno.EPERM, "simulated restricted system call")

    monkeypatch.setattr(ltiff_cli, "ProcessPoolExecutor", _DeniedExecutor)
    fallback_permission_output = tmp_path / "fallback_permission_output"
    fallback_permission_args = ltiff.parse_args(
        [str(input_dir), str(fallback_permission_output), "--workers", "2", "--no-progress"]
    )
    with caplog.at_level("WARNING", logger="luxury_tiff_batch_processor"):
        fallback_permission_processed = ltiff.run_pipeline(fallback_permission_args)
    fallback_permission_hashes = _collect_output_hashes(fallback_permission_output)

    monkeypatch.setattr(ltiff_cli, "ProcessPoolExecutor", _DeniedOSErrorExecutor)
    fallback_oserror_output = tmp_path / "fallback_oserror_output"
    fallback_oserror_args = ltiff.parse_args([str(input_dir), str(fallback_oserror_output), "--workers", "2", "--no-progress"])
    with caplog.at_level("WARNING", logger="luxury_tiff_batch_processor"):
        fallback_oserror_processed = ltiff.run_pipeline(fallback_oserror_args)
    fallback_oserror_hashes = _collect_output_hashes(fallback_oserror_output)

    assert serial_processed == parallel_processed == fallback_permission_processed == fallback_oserror_processed == 3
    assert serial_hashes == parallel_hashes == fallback_permission_hashes == fallback_oserror_hashes
    assert "Falling back to single-process execution." in caplog.text


def test_run_pipeline_fallback_avoids_parallel_progress_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _create_sample_image(input_dir / "frame.tif")

    progress_calls: list[dict[str, Any]] = []

    def _progress_spy(iterable, *, total=None, description=None, enabled=True):  # noqa: ANN001, ANN202
        progress_calls.append(
            {
                "total": total,
                "description": description,
                "enabled": enabled,
            }
        )
        yield from iterable

    class _DeniedExecutor:
        def __init__(self, max_workers: int):  # pylint: disable=unused-argument
            raise OSError(errno.EPERM, "simulated restricted system call")

    monkeypatch.setattr(ltiff_cli, "_wrap_with_progress", _progress_spy)
    monkeypatch.setattr(ltiff_cli, "ProcessPoolExecutor", _DeniedExecutor)

    args = ltiff.parse_args([str(input_dir), str(output_dir), "--workers", "2"])
    processed = ltiff.run_pipeline(args)

    assert processed == 1
    assert len(progress_calls) == 1
    assert progress_calls[0]["total"] == 1
    assert progress_calls[0]["enabled"] is True


def test_run_pipeline_submit_time_fallback_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(3):
        _create_sample_image(input_dir / f"frame_{index}.tif")

    class _SubmitDeniedExecutor:
        def __init__(self, max_workers: int):  # pylint: disable=unused-argument
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: ANN001, ANN201
            return False

        def submit(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise OSError(errno.EPERM, "simulated worker spawn restriction")

    serial_output = tmp_path / "serial_output"
    serial_args = ltiff.parse_args([str(input_dir), str(serial_output), "--workers", "1", "--no-progress"])
    serial_processed = ltiff.run_pipeline(serial_args)
    serial_hashes = _collect_output_hashes(serial_output)

    monkeypatch.setattr(ltiff_cli, "ProcessPoolExecutor", _SubmitDeniedExecutor)
    fallback_output = tmp_path / "fallback_output"
    fallback_args = ltiff.parse_args([str(input_dir), str(fallback_output), "--workers", "2", "--no-progress"])
    with caplog.at_level("WARNING", logger="luxury_tiff_batch_processor"):
        fallback_processed = ltiff.run_pipeline(fallback_args)
    fallback_hashes = _collect_output_hashes(fallback_output)

    assert serial_processed == fallback_processed == 3
    assert serial_hashes == fallback_hashes
    assert "worker startup unavailable" in caplog.text
    assert "Falling back to single-process execution." in caplog.text


def test_run_pipeline_invokes_progress_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    sample = input_dir / "frame.tif"
    _create_sample_image(sample)

    args = ltiff.parse_args([str(input_dir), str(output_dir), "--dry-run"])

    calls: Dict[str, Any] = {"called": False, "items": []}

    def stub_progress(iterable, *, total=None, description=None):
        calls["called"] = True
        calls["total"] = total
        calls["description"] = description
        for item in iterable:
            calls["items"].append(item)
            yield item

    monkeypatch.setattr(pipeline, "_PROGRESS_WRAPPER", stub_progress)

    processed = ltiff.run_pipeline(args)

    assert processed == 0
    assert calls["called"] is True
    assert calls["total"] == 1
    assert calls["items"] == [sample]
    assert "Processing" in (calls["description"] or "")


def test_run_pipeline_no_progress_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    _create_sample_image(input_dir / "frame.tif")

    args = ltiff.parse_args([str(input_dir), str(output_dir), "--dry-run", "--no-progress"])

    calls = {"called": False}

    def stub_progress(iterable, *, total=None, description=None):
        calls["called"] = True
        yield from iterable

    monkeypatch.setattr(pipeline, "_PROGRESS_WRAPPER", stub_progress)

    ltiff.run_pipeline(args)

    assert calls["called"] is False


def test_run_pipeline_supports_legacy_resize_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    sample = input_dir / "frame.tif"
    _create_sample_image(sample)

    args = ltiff.parse_args([str(input_dir), str(output_dir), "--dry-run"])
    # Simulate older integrations that provided ``resize_target`` but not ``resize_long_edge``.
    setattr(args, "resize_target", 2)
    delattr(args, "resize_long_edge")

    progress_calls = {"count": 0}

    def stub_progress(iterable, *, total=None, description=None):
        progress_calls["count"] += 1
        yield from iterable

    monkeypatch.setattr(pipeline, "_PROGRESS_WRAPPER", stub_progress)

    processed = ltiff.run_pipeline(args)

    assert processed == 0
    # Ensure we still invoked progress with the expected number of items.
    assert progress_calls["count"] == 1


@documents("Filesystem discovery respects operator scope selections")
def test_collect_images_handles_recursive(tmp_path: Path):
    input_root = tmp_path
    (input_root / "top.tif").write_bytes(b"top")
    (input_root / "upper.TIFF").write_bytes(b"upper")
    nested = input_root / "nested"
    nested.mkdir()
    (nested / "inner.tif").write_bytes(b"inner")
    (nested / "ignore.jpg").write_bytes(b"jpg")

    non_recursive = sorted(p.relative_to(input_root) for p in ltiff.collect_images(input_root, recursive=False))
    assert non_recursive == [
        Path("top.tif"),
        Path("upper.TIFF"),
    ]

    recursive = sorted(p.relative_to(input_root) for p in ltiff.collect_images(input_root, recursive=True))
    assert recursive == [
        Path("nested/inner.tif"),
        Path("top.tif"),
        Path("upper.TIFF"),
    ]


def test_parse_args_sets_default_output(tmp_path: Path):
    input_dir = tmp_path / "SV-Stills"
    input_dir.mkdir()

    args = ltiff.parse_args([str(input_dir)])

    assert args.output == input_dir.parent / "SV-Stills_lux"


def test_parse_args_workers_default_and_override(tmp_path: Path):
    input_dir = tmp_path / "source"
    input_dir.mkdir()

    args_default = ltiff.parse_args([str(input_dir)])
    assert args_default.workers == 1

    args_override = ltiff.parse_args([str(input_dir), "--workers", "4"])
    assert args_override.workers == 4


def test_run_pipeline_rejects_nested_output(tmp_path: Path):
    input_dir = tmp_path / "input"
    nested_output = input_dir / "lux"
    nested_output.mkdir(parents=True)

    args = ltiff.parse_args([str(input_dir), str(nested_output)])

    with pytest.raises(SystemExit) as excinfo:
        ltiff.run_pipeline(args)

    assert "Output folder cannot be located inside the input folder" in str(excinfo.value)


def test_run_pipeline_rejects_input_nested_in_output(tmp_path: Path):
    output_dir = tmp_path / "output"
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True)

    args = ltiff.parse_args([str(input_dir), str(output_dir)])

    with pytest.raises(SystemExit) as excinfo:
        ltiff.run_pipeline(args)

    assert "Input folder cannot be located inside the output folder" in str(excinfo.value)


@documents("User overrides cascade atop curated presets without drift")
def test_build_adjustments_applies_overrides(tmp_path: Path):
    args = ltiff.parse_args(
        [
            str(tmp_path / "input"),
            str(tmp_path / "output"),
            "--preset",
            "signature",
            "--exposure",
            "0.5",
            "--white-balance-temp",
            "7000",
            "--clarity",
            "0.33",
        ]
    )

    adjustments = ltiff.build_adjustments(args)

    assert adjustments.exposure == 0.5
    assert adjustments.white_balance_temp == 7000
    assert adjustments.clarity == 0.33
    # Ensure we still carry over preset defaults for untouched controls
    assert adjustments.shadow_lift == ltiff.LUXURY_PRESETS["signature"].shadow_lift


def test_adjustment_settings_validation_accepts_reasonable_ranges():
    settings = ltiff.AdjustmentSettings(
        exposure=1.5,
        white_balance_temp=6500,
        white_balance_tint=10,
        shadow_lift=0.3,
        highlight_recovery=0.6,
        midtone_contrast=0.2,
        vibrance=0.4,
        saturation=-0.2,
        clarity=0.1,
        chroma_denoise=0.5,
        glow=0.2,
    )

    assert settings.exposure == pytest.approx(1.5)
    assert settings.saturation == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exposure": 6.0},
        {"shadow_lift": -0.1},
        {"shadow_lift": 1.5},
        {"chroma_denoise": 1.1},
        {"glow": -0.5},
        {"white_balance_temp": 40000},
    ],
)
def test_adjustment_settings_validation_rejects_invalid_ranges(kwargs: Dict[str, float] | Dict[str, int]):
    with pytest.raises(ValueError):
        ltiff.AdjustmentSettings(**kwargs)


def test_build_adjustments_rejects_invalid_override(tmp_path: Path):
    args = ltiff.parse_args(
        [
            str(tmp_path / "input"),
            str(tmp_path / "output"),
            "--preset",
            "signature",
            "--shadow-lift",
            "-0.5",
        ]
    )

    with pytest.raises(ValueError):
        ltiff.build_adjustments(args)


@documents("Golden Hour Courtyard preset translates coastal warm scene guidance into defaults")
def test_golden_hour_courtyard_preset_matches_material_response_brief():
    preset = ltiff.LUXURY_PRESETS["golden_hour_courtyard"]

    assert preset.exposure == pytest.approx(0.08)
    assert preset.white_balance_temp == pytest.approx(5600)
    assert preset.shadow_lift == pytest.approx(0.24)
    assert preset.highlight_recovery == pytest.approx(0.18)
    assert preset.midtone_contrast == pytest.approx(0.10)
    assert preset.vibrance == pytest.approx(0.28)
    assert preset.clarity == pytest.approx(0.20)
    assert preset.glow == pytest.approx(0.12)


@documents("Platform intelligence supersedes forced uniformity")
def test_image_roundtrip_uint16_with_alpha():
    data = np.array(
        [
            [[0, 32768, 65535, 65535], [65535, 0, 32768, 0]],
            [[16384, 49152, 1024, 32768], [8192, 16384, 24576, 16384]],
        ],
        dtype=np.uint16,
    )
    # Note: PIL doesn't support uint16 RGBA directly via fromarray
    # Convert to uint8 first, as PIL will do this anyway
    data_uint8 = (data / 256).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        image = Image.fromarray(data_uint8, mode="RGBA")

    base_float, base_dtype, alpha, base_channels = ltiff.image_to_float(image, return_format="tuple4")
    assert base_float.dtype == np.float32
    # PIL converts uint16 RGBA to uint8, so base_dtype will be uint8
    assert base_dtype == np.uint8
    assert alpha is not None and alpha.shape == data.shape[:2]
    assert base_channels == 3

    restored = ltiff.float_to_dtype_array(base_float, base_dtype, alpha, base_channels)
    assert restored.dtype == np.uint8
    assert restored.shape == data.shape
    # Since PIL downcast the data to uint8, the restored data will match the PIL conversion
    expected_uint8 = np.array(image)
    np.testing.assert_array_equal(restored, expected_uint8)
