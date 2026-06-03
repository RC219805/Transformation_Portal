"""Linear light decoder for Spatial AI Foundation (Research/Training Only).

This decoder outputs float32 linear light with gamma=1.0 (NOT display-ready).

WARNING: DO NOT use for rendering pipelines.
For rendering, use: transformation_portal.lux_depth_v3.raw_loader

Architecture (ADR-023, ADR-026, Issue #890 Phase I):
- Complete isolation from rendering decode logic
- Linear gamma enforcement (gamma=1.0, no baked curves)
- Float32 HDR preservation (values >1.0 allowed)
- Full provenance tracking (EXIF + ingest metadata + transform chain)
- Contract validation (SpatialCaptureV1)
- Hard failure guardrails (no silent 8-bit collapse)

Supported Formats:
- TIFF (16-bit/32-bit, uncompressed or LZW)
- PNG (16-bit)
- EXR (32-bit float, HDR)
- RAW formats (CR2, NEF, ARW, DNG) via rawpy/LibRaw

Example:
    >>> decoder = LinearDecoder(gamma=1.0, bit_depth=32, strict_ingest=True)
    >>> result = decoder.decode("scene.CR2", emit_exr=True, emit_provenance=True)
    >>> assert result.linear_rgb.dtype == np.float32
    >>> assert result.gamma == 1.0
    >>> assert result.linear_rgb.max() > 1.0  # HDR preserved
    >>> assert Path(result.output_exr_path).exists()
    >>> assert Path(result.provenance_path).exists()
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ...core.raw_runtime import run_raw_worker
from ...ingest.canonical_json import dump_json
from .exceptions import BitDepthViolationError, ColorSpaceError, UnsupportedFormatError
from .provenance import ProvenanceCapture
from .telemetry import IngestTelemetry, NullTelemetry
from .validators import validate_bit_depth, validate_linear_output

logger = logging.getLogger(__name__)


def _canonical_f64_list(arr: np.ndarray) -> list:
    """Canonicalize array to a stable float64 list for fingerprint payloads.

    Rules:
    - Cast to float64 (no float32 precision variance).
    - Flatten row-major (C order) — (3, 3) and (9,) produce the same list.
    - Normalize -0.0 → 0.0 (platform-safe equality).
    """
    a = np.asarray(arr, dtype=np.float64).ravel(order="C").copy()
    a[a == 0.0] = 0.0  # coerce -0.0 to +0.0
    return a.tolist()


def _compute_ingest_fingerprint(
    *,
    wb: "Optional[np.ndarray]",
    black_level: "Optional[np.ndarray]",
    color_matrix: "Optional[np.ndarray]",
    raw_shape: "Tuple[int, int]",
) -> str:
    """Compute a deterministic SHA-256 provenance fingerprint for RAW ingest.

    The fingerprint encodes the validated ingest-critical metadata in a
    schema-versioned, canonicalized JSON payload before hashing.  It is stable
    under equivalent numeric representations (e.g. (3,3) vs flattened matrix,
    -0.0 vs 0.0) and across repeated executions on identical inputs.

    Schema version ``v=1`` payload keys (sorted for determinism):
    - ``v``: schema version integer (1)
    - ``raw_shape``: [height, width] as integers
    - ``wb``: float64 list or null
    - ``black_level``: float64 list or null
    - ``color_matrix``: float64 flattened list or null

    Args:
        wb: Validated camera_whitebalance array, or None.
        black_level: Validated black_level_per_channel array, or None.
        color_matrix: Selected color matrix (post _select_valid_color_matrix), or None.
        raw_shape: (height, width) from the postprocessed visible frame.

    Returns:
        Lowercase hex SHA-256 digest string (64 characters).
    """
    payload = {
        "v": 1,
        "raw_shape": [int(raw_shape[0]), int(raw_shape[1])],
        "wb": None if wb is None else _canonical_f64_list(wb),
        "black_level": None if black_level is None else _canonical_f64_list(black_level),
        "color_matrix": None if color_matrix is None else _canonical_f64_list(color_matrix),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class LinearIngestResult:
    """Result from linear light ingest decoding.

    Attributes:
        linear_rgb: Float32 linear RGB array (H, W, 3), values in [0, ∞).
        gamma: Gamma value used for decode (must be 1.0 for linear).
        bit_depth: Bit depth of output (32 for float32).
        dtype: NumPy dtype string ("float32").
        input_size: Original image dimensions (height, width).
        input_path: Path to input file.
        input_format: Input format detected (e.g., "TIFF", "PNG", "EXR").
        color_space: Color space of linear RGB output (e.g., "linear_sRGB", "camera_native_linear").
        output_exr_path: Path to output EXR file (if emit_exr=True).
        provenance_path: Path to provenance JSON (if emit_provenance=True).
        provenance_data: Provenance metadata dict.
        content_hash: SHA-256 hash of linear_rgb array.
        ingest_fingerprint: Deterministic SHA-256 of validated ingest metadata
            (wb gains, black level, selected color matrix, RAW shape). Present for
            RAW formats only; None for TIFF/PNG/EXR inputs.
    """

    linear_rgb: np.ndarray
    gamma: float
    bit_depth: int
    dtype: str
    input_size: Tuple[int, int]
    input_path: Path
    input_format: str
    color_space: str
    output_exr_path: Optional[Path] = None
    provenance_path: Optional[Path] = None
    provenance_data: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    ingest_fingerprint: Optional[str] = None

    def __post_init__(self):
        """Validate result contract."""
        # Gamma enforcement (SpatialCaptureV1 contract)
        if abs(self.gamma - 1.0) > 1e-6:
            raise ValueError(
                f"Linear ingest requires gamma=1.0, got {self.gamma}. " "This violates the SpatialCaptureV1 contract."
            )

        # Dtype enforcement
        if self.linear_rgb.dtype != np.float32:
            raise ValueError(
                f"Linear ingest requires float32 dtype, got {self.linear_rgb.dtype}. "
                "This violates the SpatialCaptureV1 contract."
            )

        # Shape validation
        if self.linear_rgb.ndim != 3 or self.linear_rgb.shape[2] != 3:
            raise ValueError(f"Linear RGB must be (H, W, 3), got shape {self.linear_rgb.shape}")

        # No 8-bit collapse enforcement
        if self.bit_depth < 32:
            logger.warning(
                f"Linear ingest should use bit_depth=32, got {self.bit_depth}. " "Lower bit depths may not preserve HDR range."
            )


class LinearDecoder:
    """Decoder for RAW/TIFF → float32 linear light (research/training only).

    This is the ONLY entry point for Spatial AI linear ingest.
    Isolated from lux_depth_v3.raw_loader per ADR-023.

    Usage:
        >>> decoder = LinearDecoder(gamma=1.0, bit_depth=32)
        >>> result = decoder.decode("scene.tiff")
        >>> # Use result.linear_rgb for training data
    """

    def __init__(
        self,
        gamma: float = 1.0,
        bit_depth: int = 32,
        strict_ingest: bool = False,
        telemetry: IngestTelemetry | None = None,
        raw_python_executable: str | None = None,
        demosaic: str = "AHD",
    ):
        """Initialize linear decoder.

        Args:
            gamma: Gamma for decode (must be 1.0 for linear).
            bit_depth: Output bit depth (32 for float32).
            strict_ingest: If True, reject 8-bit inputs to prevent lossy normalization.
                For research/training workflows requiring true linear preservation,
                set strict_ingest=True to enforce >=16-bit inputs only.
            telemetry: Optional telemetry implementation for ingest boundary instrumentation.
                Defaults to NullTelemetry (zero overhead).
            demosaic: rawpy.DemosaicAlgorithm name applied during RAW postprocess.
                Default "AHD" preserves prior behavior. Unknown names fail closed.

        Raises:
            ValueError: If gamma != 1.0 (linear ingest contract).
        """
        if abs(gamma - 1.0) > 1e-6:
            raise ValueError(
                f"Linear ingest requires gamma=1.0 (got {gamma}). "
                "This is a non-negotiable contract for research/training data."
            )

        self.gamma = gamma
        self.bit_depth = bit_depth
        self.strict_ingest = strict_ingest
        self._telemetry: IngestTelemetry = telemetry or NullTelemetry()
        self._raw_python_executable = raw_python_executable
        self.demosaic = str(demosaic)

    def _emit_telemetry(self, event: str, **fields: object) -> None:
        """Best-effort telemetry emission that never interrupts ingest flow."""
        try:
            self._telemetry.emit(event, **fields)
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
            logger.warning(
                "Ignoring ingest telemetry backend failure for event '%s'",
                event,
                exc_info=True,
            )

    def decode(
        self,
        input_path: Path | str,
        output_dir: Optional[Path | str] = None,
        emit_exr: bool = False,
        emit_provenance: bool = False,
    ) -> LinearIngestResult:
        """Decode image to float32 linear light.

        Args:
            input_path: Path to input file (TIFF, PNG, EXR).
            output_dir: Output directory for EXR and provenance files.
                If None, uses input file's directory.
            emit_exr: Save linear RGB as EXR (float32 HDR).
            emit_provenance: Save provenance metadata as JSON.

        Returns:
            LinearIngestResult with linear RGB and metadata.

        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If input format not supported.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Linear ingest: {input_path}")

        # Detect format
        format_str = self._detect_format(input_path)
        logger.debug(f"Detected format: {format_str}")

        if format_str.startswith("RAW_") and self._raw_python_executable is not None:
            linear_rgb, input_size, ingest_fingerprint, color_space = self._decode_raw_via_subprocess(input_path)
        else:
            # Detect color space (especially for RAW files)
            if format_str.startswith("RAW_"):
                color_space = self._detect_raw_color_space(input_path)
            else:
                # Non-RAW formats: assume linear_sRGB for Phase I
                # (TIFF/PNG/EXR don't have embedded color matrices like RAW)
                color_space = "linear_sRGB"
            logger.debug(f"Color space: {color_space}")

            # Decode to linear RGB
            linear_rgb, input_size, ingest_fingerprint = self._decode_linear(input_path, format_str)

        # Compute content hash
        content_hash = self._compute_content_hash(linear_rgb)

        # Build provenance
        provenance = self._build_provenance(
            input_path=input_path,
            format_str=format_str,
            input_size=input_size,
            content_hash=content_hash,
        )

        # Build result
        result = LinearIngestResult(
            linear_rgb=linear_rgb,
            gamma=self.gamma,
            bit_depth=self.bit_depth,
            dtype="float32",
            input_size=input_size,
            input_path=input_path,
            input_format=format_str,
            color_space=color_space,
            provenance_data=provenance,
            content_hash=content_hash,
            ingest_fingerprint=ingest_fingerprint,
        )

        # Emit artifacts
        if emit_exr:
            result.output_exr_path = self._save_exr(linear_rgb, output_dir, input_path.stem)

        if emit_provenance:
            if ingest_fingerprint is not None:
                provenance["ingest_fingerprint"] = ingest_fingerprint
            result.provenance_path = self._save_provenance(provenance, output_dir, input_path.stem)

        logger.info(
            f"Linear ingest complete: {input_size[1]}x{input_size[0]}, "
            f"range=[{linear_rgb.min():.3f}, {linear_rgb.max():.3f}], "
            f"hash={content_hash[:16]}" + (f", fingerprint={ingest_fingerprint[:16]}" if ingest_fingerprint else "")
        )

        return result

    def _decode_raw_via_subprocess(
        self,
        path: Path,
    ) -> Tuple[np.ndarray, Tuple[int, int], Optional[str], str]:
        """Decode RAW via an isolated subprocess runtime."""
        if self._raw_python_executable is None:
            raise RuntimeError("RAW subprocess decode requested without a Python executable.")

        array, metadata = run_raw_worker(
            python_executable=self._raw_python_executable,
            command_name="linear_decode",
            input_path=path,
            payload={
                "gamma": self.gamma,
                "bit_depth": self.bit_depth,
                "strict_ingest": self.strict_ingest,
                "demosaic": self.demosaic,
            },
            start=Path(__file__),
        )
        linear_rgb = np.asarray(array, dtype=np.float32)
        size_payload = metadata.get("input_size")
        if not isinstance(size_payload, list) or len(size_payload) != 2:
            raise RuntimeError(f"RAW worker returned invalid input_size metadata: {size_payload!r}")
        input_size = (int(size_payload[0]), int(size_payload[1]))
        color_space = str(metadata.get("color_space") or "linear_sRGB")
        ingest_fingerprint = metadata.get("ingest_fingerprint")
        if ingest_fingerprint is not None:
            ingest_fingerprint = str(ingest_fingerprint)
        return linear_rgb, input_size, ingest_fingerprint, color_space

    def _detect_format(self, path: Path) -> str:
        """Detect image format from file extension.

        Args:
            path: Input file path.

        Returns:
            Format string (TIFF, PNG, EXR, RAW_CR2, RAW_NEF, RAW_ARW, RAW_DNG).

        Raises:
            UnsupportedFormatError: If format not supported.
        """
        ext = path.suffix.lower()
        if ext in [".tif", ".tiff"]:
            return "TIFF"
        elif ext == ".png":
            return "PNG"
        elif ext == ".exr":
            return "EXR"
        elif ext == ".cr2":
            return "RAW_CR2"
        elif ext == ".nef":
            return "RAW_NEF"
        elif ext == ".arw":
            return "RAW_ARW"
        elif ext == ".dng":
            return "RAW_DNG"
        elif ext in [".jpg", ".jpeg"]:
            raise UnsupportedFormatError(
                input_path=path,
                detected_format="JPEG",
                supported_formats=[
                    "TIFF (16-bit/32-bit)",
                    "PNG (16-bit)",
                    "EXR (32-bit float)",
                    "RAW (CR2, NEF, ARW, DNG)",
                ],
            )
        else:
            raise UnsupportedFormatError(
                input_path=path,
                detected_format=ext,
            )

    def _decode_linear(self, path: Path, format_str: str) -> Tuple[np.ndarray, Tuple[int, int], Optional[str]]:
        """Decode image to float32 linear RGB.

        Args:
            path: Input file path.
            format_str: Format string from _detect_format.

        Returns:
            Tuple of (linear_rgb array, (height, width), ingest_fingerprint).
            ingest_fingerprint is a SHA-256 hex string for RAW formats; None otherwise.

        Raises:
            BitDepthViolationError: If strict_ingest validation fails.
            RuntimeError: If decode fails.
        """
        try:
            if format_str == "EXR":
                linear_rgb, size = self._decode_exr(path)
                return linear_rgb, size, None
            elif format_str.startswith("RAW_"):
                return self._decode_raw(path, format_str)
            else:
                linear_rgb, size = self._decode_pillow(path, format_str)
                return linear_rgb, size, None
        except (BitDepthViolationError, UnsupportedFormatError):
            # Let custom exceptions propagate directly
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to decode {path}: {e}") from e

    def _decode_exr(self, path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Decode EXR using OpenEXR (if available) or Pillow fallback.

        Args:
            path: Path to EXR file.

        Returns:
            Tuple of (linear_rgb array, (height, width)).
        """
        try:
            import Imath
            import OpenEXR

            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header()

            # Get dimensions
            dw = header["dataWindow"]
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Read RGB channels
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            r_str = exr_file.channel("R", pt)
            g_str = exr_file.channel("G", pt)
            b_str = exr_file.channel("B", pt)

            # Convert to numpy
            r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
            g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
            b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)

            linear_rgb = np.stack([r, g, b], axis=-1)

            return linear_rgb, (height, width)

        except ImportError:
            logger.warning("OpenEXR not available, using Pillow fallback (may be slower)")
            return self._decode_pillow(path, "EXR")

    def _decode_pillow(self, path: Path, format_str: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Decode image using Pillow or tifffile (TIFF, PNG, EXR fallback).

        Args:
            path: Input file path.
            format_str: Format string.

        Returns:
            Tuple of (linear_rgb array, (height, width)).

        Raises:
            BitDepthViolationError: If strict_ingest=True and input is 8-bit.
        """
        # Use tifffile for TIFF to preserve 16-bit
        if format_str == "TIFF":
            try:
                import tifffile

                img_array = tifffile.imread(str(path))
            except ImportError:
                # Fallback to PIL
                logger.warning("tifffile not available, using PIL (may lose bit depth)")
                img = Image.open(path)
                img_array = np.array(img)
        else:
            img = Image.open(path)
            img_array = np.array(img)

        # Convert to RGB if grayscale
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            # Drop alpha channel
            img_array = img_array[:, :, :3]

        # Validate bit depth using new validator
        validate_bit_depth(
            input_path=path,
            array=img_array,
            min_bits=16,
            strict=self.strict_ingest,
        )

        # Convert to float32 and normalize to [0, 1] range initially
        if img_array.dtype == np.uint8:
            linear_rgb = img_array.astype(np.float32) / 255.0
        elif img_array.dtype == np.uint16:
            linear_rgb = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype in [np.float32, np.float64]:
            linear_rgb = img_array.astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {img_array.dtype}")

        # Note: This is a simplified linear decode.
        # Full RAW decode with LibRaw (via rawpy) handles:
        # - Demosaicing
        # - White balance
        # - Color matrix transforms
        # - Lens corrections
        #
        # For TIFF/PNG/EXR inputs, we assume already demosaiced/color-corrected.

        height, width = img_array.shape[:2]
        return linear_rgb, (height, width)

    def _decode_raw(self, path: Path, format_str: str) -> Tuple[np.ndarray, Tuple[int, int], Optional[str]]:
        """Decode RAW image using rawpy (LibRaw wrapper).

        This method provides proper RAW decode with:
        - Linear demosaicing (no gamma)
        - Configurable white balance
        - No output curve baking
        - 16-bit → float32 pipeline

        Args:
            path: Path to RAW file.
            format_str: RAW format string (RAW_CR2, RAW_NEF, etc.).

        Returns:
            Tuple of (linear_rgb array, (height, width), ingest_fingerprint).
            ingest_fingerprint is a SHA-256 hex string of validated ingest metadata.
        """
        try:
            import rawpy
        except ImportError as e:
            raise ImportError(
                f"RAW format {format_str} requires rawpy package. "
                "Use `./scripts/setup/install_raw_runtime.sh` for the isolated RAW runtime, "
                "or deliberately install the RAW extra into this active interpreter for development."
            ) from e

        try:
            with rawpy.imread(str(path)) as raw:
                # Validate RAW metadata before postprocess
                self._validate_raw_metadata(raw)

                # Decode with linear settings
                # Key parameters for linear ingest:
                # - output_color=rawpy.ColorSpace.sRGB: Output in linear sRGB (Phase I)
                # - gamma=(1,1): NO gamma correction (linear light)
                # - no_auto_bright=True: No auto brightness adjustment
                # - output_bps=16: 16-bit output (max precision before float32)
                # - use_camera_wb=True: Use camera white balance (default)
                # - demosaic_algorithm: Half-size for speed, can upgrade to AHD for quality

                from ...core.raw_runtime import resolve_demosaic_algorithm

                demosaic_enum = resolve_demosaic_algorithm(self.demosaic)

                rgb = raw.postprocess(
                    gamma=(1, 1),  # Linear gamma (no correction)
                    no_auto_bright=True,  # No auto exposure
                    output_color=rawpy.ColorSpace.sRGB,  # Linear sRGB color space
                    output_bps=16,  # 16-bit output
                    use_camera_wb=True,  # Use camera white balance from EXIF
                    half_size=False,  # Full resolution
                    four_color_rgb=False,  # Standard 3-color RGB
                    demosaic_algorithm=demosaic_enum,
                    median_filter_passes=0,  # No median filtering (preserve detail)
                    use_auto_wb=False,  # Don't override camera WB
                    highlight_mode=rawpy.HighlightMode.Clip,  # Clip highlights (no reconstruction)
                )

                # Guard: postprocess must return uint16 (H, W, 3)
                if rgb.dtype != np.uint16:
                    self._emit_telemetry("ingest.postprocess_guard_failed", reason="dtype_mismatch", dtype=str(rgb.dtype))
                    raise RuntimeError(
                        f"RAW decode: expected uint16 from postprocess (output_bps=16), "
                        f"got {rgb.dtype}. Cannot normalize safely."
                    )
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    self._emit_telemetry("ingest.postprocess_guard_failed", reason="shape_mismatch", shape=tuple(rgb.shape))
                    raise RuntimeError(f"RAW decode: expected (H, W, 3) from postprocess, got shape {rgb.shape}.")

                # Convert uint16 [0, 65535] to float32 [0, 1]
                linear_rgb = rgb.astype(np.float32) / 65535.0

                # Note: rawpy returns (H, W, C) RGB directly (already demosaiced)
                height, width = linear_rgb.shape[:2]

                logger.info(
                    f"RAW decode complete: {format_str}, {width}x{height}, "
                    f"range=[{linear_rgb.min():.4f}, {linear_rgb.max():.4f}]"
                )

                # Compute deterministic ingest provenance fingerprint from
                # validated metadata (wb, bl, selected color matrix, shape).
                wb_for_fp = (
                    np.asarray(raw.camera_whitebalance, dtype=np.float64)
                    if hasattr(raw, "camera_whitebalance") and raw.camera_whitebalance is not None
                    else None
                )
                bl_for_fp = (
                    np.asarray(raw.black_level_per_channel, dtype=np.float64)
                    if hasattr(raw, "black_level_per_channel") and raw.black_level_per_channel is not None
                    else None
                )
                cm_for_fp = self._select_valid_color_matrix(
                    raw.color_matrix if hasattr(raw, "color_matrix") else None,
                    raw.rgb_xyz_matrix if hasattr(raw, "rgb_xyz_matrix") else None,
                )
                fingerprint = _compute_ingest_fingerprint(
                    wb=wb_for_fp,
                    black_level=bl_for_fp,
                    color_matrix=cm_for_fp,
                    raw_shape=(height, width),
                )

                return linear_rgb, (height, width), fingerprint

        except Exception as e:
            # ValueError from _validate_raw_metadata must propagate unchanged so
            # callers can distinguish metadata failures. RuntimeError from our
            # local guards uses a stable "RAW decode:" prefix and should also
            # propagate directly. All other failures are wrapped with filename
            # context for diagnostics.
            if isinstance(e, ValueError):
                raise
            if isinstance(e, RuntimeError) and str(e).startswith("RAW decode:"):
                raise
            raise RuntimeError(f"Failed to decode RAW file {path.name}: {e}") from e

    def _detect_raw_color_space(self, path: Path) -> str:
        """Detect and validate color space from RAW file metadata.

        This method extracts camera color matrix from RAW metadata to determine
        the explicit color space of the linear RGB output. Phase I hardcodes
        linear sRGB via rawpy.ColorSpace.sRGB in _decode_raw().

        Args:
            path: Path to RAW file.

        Returns:
            Color space string (e.g., "camera_native_linear", "linear_sRGB").

        Raises:
            ColorSpaceError: If color matrix is missing or invalid.
            ImportError: If rawpy is not installed.
        """
        try:
            import rawpy
        except ImportError as e:
            raise ImportError(
                "RAW color space detection requires rawpy package. "
                "Use `./scripts/setup/install_raw_runtime.sh` for the isolated RAW runtime, "
                "or deliberately install the RAW extra into this active interpreter for development."
            ) from e

        try:
            with rawpy.imread(str(path)) as raw:
                # Extract camera color matrices
                # LibRaw provides color matrix data via rawpy
                # Check for both color_matrix (camera) and rgb_xyz_matrix (standard)

                # rawpy exposes raw.color_matrix and raw.rgb_xyz_matrix
                # color_matrix: camera-specific color transformation
                # rgb_xyz_matrix: RGB to XYZ transformation

                color_matrix = None
                rgb_xyz_matrix = None

                # Try to get color matrices
                if hasattr(raw, "color_matrix"):
                    color_matrix = raw.color_matrix
                if hasattr(raw, "rgb_xyz_matrix"):
                    rgb_xyz_matrix = raw.rgb_xyz_matrix

                # Select the best available matrix (_select_valid_color_matrix is
                # the single validation authority: length, NaN, norm, fallback).
                matrix_to_check = self._select_valid_color_matrix(color_matrix, rgb_xyz_matrix)

                if matrix_to_check is None:
                    # Build diagnostic detail for forensic debugging
                    def _diag(m: Any) -> str:
                        if m is None:
                            return "absent"
                        try:
                            arr = np.asarray(m, dtype=np.float64)
                        except Exception as exc:
                            return f"unparseable(type={type(m).__name__}, error={exc})"
                        try:
                            norm = np.linalg.norm(arr) if arr.size > 0 else 0.0
                        except Exception as exc:
                            return f"shape={getattr(arr, 'shape', '?')} norm=uncomputable(error={exc})"
                        return f"shape={arr.shape} norm={norm:.3e}"

                    raise ColorSpaceError(
                        input_path=path,
                        reason=(
                            "No valid camera color matrix found in RAW metadata. "
                            f"color_matrix={_diag(color_matrix)}, "
                            f"rgb_xyz_matrix={_diag(rgb_xyz_matrix)}. "
                            "Both were rejected (wrong shape, NaN/Inf, or norm < 1e-6)."
                        ),
                        matrix_present=color_matrix is not None or rgb_xyz_matrix is not None,
                    )

                # Phase I: We use rawpy.ColorSpace.sRGB in _decode_raw()
                # This produces linear sRGB output (gamma=1.0)
                # Future phases may expose camera_native_linear option
                return "linear_sRGB"

        except ColorSpaceError:
            # Let ColorSpaceError propagate
            raise
        except Exception as e:
            raise ColorSpaceError(
                input_path=path,
                reason=f"Failed to read RAW metadata: {e}",
                matrix_present=False,
            ) from e

    def _validate_raw_metadata(self, raw: Any) -> None:
        """Validate RAW file metadata before decode to prevent silent numeric corruption.

        Checks white balance gains, black level, and output shape safety.
        Raises ValueError with a clear message on the first detected issue.

        Args:
            raw: rawpy RawPy object (open file context).

        Raises:
            ValueError: If metadata is malformed (non-finite WB gains, wrong
                        channel count, negative black level, etc.).
        """
        # --- White Balance Gains ---
        # camera_whitebalance: [R, G1, B, G2] multipliers from camera EXIF
        if hasattr(raw, "camera_whitebalance"):
            wb = raw.camera_whitebalance
            if wb is not None:
                try:
                    wb_arr = np.array(wb, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    self._emit_telemetry("ingest.validation_failed", field="camera_whitebalance", reason="non_numeric")
                    raise ValueError(
                        "RAW metadata: camera_whitebalance is unparseable to float64. "
                        f"type={type(wb).__name__}, value={wb!r}, error={exc}"
                    ) from exc
                if wb_arr.size == 0:
                    self._emit_telemetry("ingest.validation_failed", field="camera_whitebalance", reason="empty")
                    raise ValueError(
                        "RAW metadata: camera_whitebalance is empty. " "Cannot decode without valid white balance gains."
                    )
                if wb_arr.size != 4:
                    self._emit_telemetry(
                        "ingest.validation_failed", field="camera_whitebalance", reason="invalid_channel_count"
                    )
                    raise ValueError(
                        f"RAW metadata: camera_whitebalance has unexpected channel count {wb_arr.size} "
                        f"(expected 4 for [R, G1, B, G2]): {wb_arr.tolist()}."
                    )
                if np.isnan(wb_arr).any():
                    self._emit_telemetry("ingest.validation_failed", field="camera_whitebalance", reason="nan")
                    raise ValueError(
                        f"RAW metadata: camera_whitebalance contains NaN values: {wb_arr}. " "Malformed camera EXIF."
                    )
                if np.isinf(wb_arr).any():
                    self._emit_telemetry("ingest.validation_failed", field="camera_whitebalance", reason="inf")
                    raise ValueError(
                        f"RAW metadata: camera_whitebalance contains infinity values: {wb_arr}. " "Malformed camera EXIF."
                    )
                if np.any(wb_arr <= 0.0):
                    # All channels (R, G1, B, G2) must be positive
                    bad_indices = np.where(wb_arr <= 0.0)[0].tolist()
                    self._emit_telemetry("ingest.validation_failed", field="camera_whitebalance", reason="non_positive")
                    raise ValueError(
                        f"RAW metadata: camera_whitebalance has zero or negative gain(s) "
                        f"at channel(s) {bad_indices}: {wb_arr.tolist()}. "
                        "Cannot apply white balance without strictly positive multipliers."
                    )

        # --- Black Level ---
        # black_level_per_channel: per-channel black point offsets
        if hasattr(raw, "black_level_per_channel"):
            bl = raw.black_level_per_channel
            if bl is not None:
                try:
                    bl_arr = np.array(bl, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    self._emit_telemetry("ingest.validation_failed", field="black_level_per_channel", reason="non_numeric")
                    raise ValueError(
                        "RAW metadata: black_level_per_channel is unparseable to float64. "
                        f"type={type(bl).__name__}, value={bl!r}, error={exc}"
                    ) from exc
                if bl_arr.size == 0:
                    self._emit_telemetry("ingest.validation_failed", field="black_level_per_channel", reason="empty")
                    raise ValueError("RAW metadata: black_level_per_channel is empty. " "Cannot safely subtract black level.")
                if np.isnan(bl_arr).any():
                    self._emit_telemetry("ingest.validation_failed", field="black_level_per_channel", reason="nan")
                    raise ValueError(f"RAW metadata: black_level_per_channel contains NaN: {bl_arr}.")
                if np.isinf(bl_arr).any():
                    self._emit_telemetry("ingest.validation_failed", field="black_level_per_channel", reason="inf")
                    raise ValueError(f"RAW metadata: black_level_per_channel contains infinity values: {bl_arr}.")
                if np.any(bl_arr < 0.0):
                    self._emit_telemetry("ingest.validation_failed", field="black_level_per_channel", reason="negative")
                    raise ValueError(
                        f"RAW metadata: black_level_per_channel has negative values: {bl_arr}. "
                        "Black level must be non-negative."
                    )
                if bl_arr.size not in (1, 3, 4):
                    self._emit_telemetry(
                        "ingest.validation_failed", field="black_level_per_channel", reason="invalid_channel_count"
                    )
                    raise ValueError(
                        f"RAW metadata: black_level_per_channel has unexpected channel count "
                        f"{bl_arr.size} (expected 1, 3, or 4 for Bayer/RGB layouts): {bl_arr.tolist()}."
                    )

        # --- Raw image shape sanity ---
        if hasattr(raw, "raw_image") and raw.raw_image is not None:
            if raw.raw_image.ndim != 2:
                self._emit_telemetry("ingest.validation_failed", field="raw_image", reason="wrong_ndim")
                raise ValueError(f"RAW metadata: raw_image expected 2D (H×W Bayer), " f"got shape {raw.raw_image.shape}.")

    def _select_valid_color_matrix(
        self,
        color_matrix: Any,
        rgb_xyz_matrix: Any,
    ) -> Optional[np.ndarray]:
        """Select the best available color matrix, preferring color_matrix over rgb_xyz_matrix.

        Fallback selection order (first valid wins):
          1. color_matrix  — camera-specific matrix from EXIF
          2. rgb_xyz_matrix — standardised RGB→XYZ matrix
          3. None           — caller must raise ColorSpaceError

        A matrix is valid if it:
          - Has shape (3, 3), (3, 4), (4, 3), or (9,)
            - (3, 4) and (4, 3) are common LibRaw/rawpy metadata layouts.
            - They are deterministically contracted to 3×3 by dropping the
              trailing channel row/column before flattening.
          - Contains no NaN or Inf values
          - Has L2 norm >= 1e-6 (rejects zero-filled and near-zero garbage)

        The norm threshold (1e-6) is intentionally conservative. A physically
        meaningful 3×3 color transform always has a norm well above this value.

        Args:
            color_matrix: Camera color matrix (may be None, (3,3) ndarray, flat list,
                          NaN-filled, near-zero, or wrong shape).
            rgb_xyz_matrix: RGB-to-XYZ matrix (fallback, same acceptance criteria).

        Returns:
            np.ndarray of shape (9,), dtype float64, or None if neither is usable.
        """

        def _normalize(matrix: Any) -> Optional[np.ndarray]:
            if matrix is None:
                return None
            try:
                arr = np.asarray(matrix, dtype=np.float64)
            except (TypeError, ValueError):
                return None
            # Accept common LibRaw/rawpy matrix layouts and reduce to 3x3.
            if arr.shape == (3, 3):
                arr = arr.reshape(9)
            elif arr.shape == (3, 4):
                # Camera -> XYZ with an extra channel column (often second green).
                # Keep first 3 columns for deterministic 3x3 canonicalization.
                arr = arr[:, :3].reshape(9)
            elif arr.shape == (4, 3):
                # Camera channel rows with an extra row (often second green).
                # Keep first 3 rows for deterministic 3x3 canonicalization.
                arr = arr[:3, :].reshape(9)
            elif arr.ndim == 1 and arr.size == 9:
                pass
            else:
                return None
            if np.isnan(arr).any():
                return None
            if np.isinf(arr).any():
                return None
            try:
                norm = np.linalg.norm(arr)
            except (TypeError, ValueError):
                return None
            if norm < 1e-6:
                return None
            return arr

        primary = _normalize(color_matrix)
        if primary is not None:
            return primary

        # color_matrix was present but invalid, falling back to rgb_xyz_matrix
        fallback = _normalize(rgb_xyz_matrix)
        if fallback is not None and color_matrix is not None:
            # Emit telemetry ONLY when color_matrix was present but invalid
            self._emit_telemetry("ingest.matrix_fallback_used", from_="color_matrix", to="rgb_xyz_matrix")
        return fallback

    def _compute_content_hash(self, array: np.ndarray) -> str:
        """Compute SHA-256 hash of array content.

        Args:
            array: NumPy array to hash.

        Returns:
            Hex string of SHA-256 hash.
        """
        hasher = hashlib.sha256()
        hasher.update(array.tobytes())
        return hasher.hexdigest()

    def _build_provenance(
        self,
        input_path: Path,
        format_str: str,
        input_size: Tuple[int, int],
        content_hash: str,
    ) -> Dict[str, Any]:
        """Build provenance metadata dict.

        Args:
            input_path: Input file path.
            format_str: Input format.
            input_size: Image dimensions (height, width).
            content_hash: SHA-256 hash of output array.

        Returns:
            Provenance dict.
        """
        return {
            "input": {
                "path": str(input_path),
                "format": format_str,
                "size": {"height": input_size[0], "width": input_size[1]},
            },
            "decode": {
                "gamma": self.gamma,
                "bit_depth": self.bit_depth,
                "dtype": "float32",
                "method": "linear_decoder_v1",
                "contract": "SpatialCaptureV1",
            },
            "output": {
                "content_hash": content_hash,
                "hash_algorithm": "sha256",
            },
            "adr": "ADR-026",
            "module": "transformation_portal.spatial_ai.ingest.linear_decoder",
        }

    def _save_exr(self, linear_rgb: np.ndarray, output_dir: Path, stem: str) -> Path:
        """Save linear RGB as EXR (float32 HDR).

        Args:
            linear_rgb: Float32 RGB array.
            output_dir: Output directory.
            stem: Output filename stem.

        Returns:
            Path to saved EXR file.

        Raises:
            RuntimeError: If OpenEXR is not available (fail-loud for HDR preservation).
        """
        output_path = output_dir / f"{stem}_linear.exr"

        try:
            import Imath
            import OpenEXR

            height, width = linear_rgb.shape[:2]
            header = OpenEXR.Header(width, height)
            header["channels"] = {
                "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            }

            exr = OpenEXR.OutputFile(str(output_path), header)
            r = linear_rgb[:, :, 0].tobytes()
            g = linear_rgb[:, :, 1].tobytes()
            b = linear_rgb[:, :, 2].tobytes()
            exr.writePixels({"R": r, "G": g, "B": b})
            exr.close()

        except ImportError as e:
            raise RuntimeError(
                "emit_exr=True requires OpenEXR package for HDR preservation. "
                "Install with: pip install OpenEXR Imath\n"
                "Refusing to silently degrade to clipped 16-bit TIFF fallback."
            ) from e

        logger.debug(f"Saved linear EXR: {output_path}")
        return output_path

    def _save_provenance(self, provenance: Dict[str, Any], output_dir: Path, stem: str) -> Path:
        """Save provenance metadata as JSON.

        Args:
            provenance: Provenance dict.
            output_dir: Output directory.
            stem: Output filename stem.

        Returns:
            Path to saved JSON file.
        """
        output_path = output_dir / f"{stem}_provenance.json"
        with open(output_path, "w", encoding="utf-8") as f:
            dump_json(
                provenance,
                f,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )

        logger.debug(f"Saved provenance: {output_path}")
        return output_path


# Convenience function for direct usage
def decode(
    input_path: Path | str,
    gamma: float = 1.0,
    bit_depth: int = 32,
    strict_ingest: bool = False,
    output_dir: Optional[Path | str] = None,
    emit_exr: bool = False,
    emit_provenance: bool = False,
    raw_python_executable: str | None = None,
    demosaic: str = "AHD",
) -> LinearIngestResult:
    """Decode image to float32 linear light (convenience function).

    Args:
        input_path: Path to input file.
        gamma: Gamma for decode (must be 1.0).
        bit_depth: Output bit depth (32 for float32).
        strict_ingest: If True, reject 8-bit inputs.
        output_dir: Output directory (defaults to input directory).
        emit_exr: Save linear RGB as EXR.
        emit_provenance: Save provenance JSON.

    Returns:
        LinearIngestResult with linear RGB and metadata.

    Example:
        >>> result = decode("scene.tiff", emit_exr=True, emit_provenance=True)
        >>> assert result.gamma == 1.0
        >>> assert result.linear_rgb.dtype == np.float32
    """
    decoder = LinearDecoder(
        gamma=gamma,
        bit_depth=bit_depth,
        strict_ingest=strict_ingest,
        raw_python_executable=raw_python_executable,
        demosaic=demosaic,
    )
    return decoder.decode(
        input_path=input_path,
        output_dir=output_dir,
        emit_exr=emit_exr,
        emit_provenance=emit_provenance,
    )
