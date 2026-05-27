"""SAM2 backend for segmentation (Phase 2.1 - Production Implementation).

This module wraps Meta's Segment Anything Model 2 (SAM2) for:
- Automatic mask generation (full image)
- Prompted segmentation (points/bboxes)
- Video temporal tracking

Architecture:
- Direct checkpoint loading (not HuggingFace Hub)
- GPU/CPU/MPS device selection
- Batched inference for efficiency
- Contract-driven input/output

Model Variants:
- sam2_hiera_base_plus: Faster, good quality
- sam2_hiera_large: Slower, best quality

Example:
    >>> backend = SAM2Backend(model_size="large", device="cuda")
    >>> from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
    >>> seg_input = SegmentationInput(
    ...     image=linear_rgb,  # (H, W, 3) float32
    ...     gamma=1.0,
    ...     mode="auto"
    ... )
    >>> result = backend.segment(seg_input)
    >>> print(f"Found {len(result.masks)} segments")

License: Apache 2.0 (commercial OK, no tier restrictions)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Literal, Optional, cast

import numpy as np
from PIL import Image

from transformation_portal.core.security import ModelLockError, resolve_model_lock_revision
from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult
from transformation_portal.spatial_ai.segmentation.metadata import make_mask_metadata, metadata_from_mask
from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig
from transformation_portal.spatial_ai.segmentation.tiling.engine import TiledSegmentationEngine
from transformation_portal.spatial_ai.segmentation.tiling.merger import BinaryUnionTileMerger
from transformation_portal.spatial_ai.segmentation.tiling.planner import UniformTilingPlanner
from transformation_portal.spatial_ai.segmentation.tiling.types import (
    BBox,
    GlobalSeedHints,
    SoftMaskPatch,
    TileInstance,
    TileSpec,
)
from transformation_portal.spatial_ai.segmentation.tiling.validator import SeamMergeValidator

logger = logging.getLogger(__name__)

_SHA256_HEX_RE = re.compile(r"^[a-fA-F0-9]{64}$")


def _compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for a file using streaming reads.

    Routes through ``_content_digest.compute_file_sha256_uncached`` so
    ``_validate_checkpoint_sha256`` reads fresh bytes on every call.
    Stat-tuple memoization cannot detect a same-size, mtime-restored,
    ctime-reset overwrite on operator-controllable filesystems, so the
    integrity path opts out of memoization entirely. Cache-key uses
    (e.g. in ``lux_depth_v3.segmentation._cache``) route through the
    memoized ``compute_file_sha256`` helper instead. ``chunk_size`` is
    accepted for backward compatibility but no longer respected; the
    shared helper uses a fixed 1 MiB chunk that matches the previous
    default. (Tracks N-3, audit finding #4.)
    """
    del chunk_size  # preserved for callers; shared helper uses 1 MiB.
    from transformation_portal.spatial_ai.segmentation._content_digest import compute_file_sha256_uncached

    return compute_file_sha256_uncached(file_path)


def _validate_sha256_hex(expected_sha256: str) -> str:
    """Normalize and validate SHA-256 hex format."""
    normalized = expected_sha256.strip().lower()
    if not _SHA256_HEX_RE.fullmatch(normalized):
        raise RuntimeError(f"Invalid SHA256 digest format: {expected_sha256!r}")
    return normalized


class SAM2CheckpointIntegrityError(RuntimeError):
    """Raised when a SAM2 checkpoint hash does not match the expected digest."""


class SAM2Backend:
    """SAM2 segmentation backend with direct checkpoint loading.

    Attributes:
        model_size: Model variant ("base" or "large").
        device: Compute device ("cuda", "cpu", "mps").
        checkpoint_path: Path to model checkpoint file.
    """

    # The upstream sam2 loader accepts both Hydra short names and config file
    # paths via the shared build_sam2(config_file=...) surface. Base
    # intentionally remains on the legacy July 2024 config until its own
    # canonical SAM 2.1 pin is carried; large migrates to the SAM 2.1 config
    # path now.
    MODEL_CONFIGS = {
        "base": "sam2_hiera_b+",
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }

    # Default checkpoint names
    DEFAULT_CHECKPOINTS = {
        "base": "sam2_hiera_base_plus.pt",
        "large": "sam2.1_hiera_large.pt",
    }

    CHECKPOINT_URLS = {
        "base": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    }

    # Checkpoint SHA-256 digests (must match downloaded artifacts exactly)
    CHECKPOINT_SHA256 = {
        "base": "d0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071",
        "large": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    }

    SUPPORTED_DEVICES = {"auto", "cuda", "cpu", "mps"}
    name = "sam2"

    def __init__(
        self,
        model_size: Literal["base", "large"] = "base",
        device: Literal["auto", "cuda", "cpu", "mps"] = "cuda",
        checkpoint_path: Optional[str] = None,
        model_config: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
        prefer_hf_pipeline: Optional[bool] = None,
        generator_kwargs: Optional[Dict[str, Any]] = None,
        enable_material_classification: bool = False,
        material_confidence_threshold: float = 0.3,
        material_classification_strict: bool = False,
        tiling: Optional[SegmentationTilingConfig] = None,
    ) -> None:
        """Initialize SAM2 backend.

        Args:
            model_size: Model variant ("base" or "large").
            device: Compute device. Accepted values are "auto", "cuda", "cpu",
                or "mps". When "auto", device selection prefers MPS > CUDA > CPU
                based on availability. Requested device falls back to an
                available alternative if unavailable.
            checkpoint_path: Path to local checkpoint file. If None and
                ``prefer_hf_pipeline`` is False (the default), a model-size
                specific default checkpoint name under ``checkpoints/`` is used
                (for example, ``checkpoints/sam2.1_hiera_large.pt``). May be None
                when repo-backed HuggingFace loading is enabled via
                ``prefer_hf_pipeline=True``.
            model_config: Optional SAM2 config override. When omitted, resolves
                to the model-size default from ``MODEL_CONFIGS``.
            expected_sha256: Optional checkpoint checksum override. When omitted,
                resolves to the built-in digest in ``CHECKPOINT_SHA256`` for the
                selected model size.
            repo_id: HuggingFace Hub repository ID for model weights
                (e.g., "facebook/sam2-hiera-large"). Required when
                ``prefer_hf_pipeline=True``.
            revision: HuggingFace revision string used when loading from the Hub.
                When ``prefer_hf_pipeline=True``, this must resolve to a pinned
                40-character commit SHA, either passed directly or via the
                model lock manifest. Branch names or other unpinned revisions
                are rejected and raise ``ValueError``.
            prefer_hf_pipeline: When True, load model weights from HuggingFace
                Hub using ``repo_id`` and ``revision`` instead of a local
                checkpoint. This path is opt-in; the default loading model
                remains checkpoint-first. If the HuggingFace load fails and a
                local checkpoint exists, it falls back to checkpoint loading.
            generator_kwargs: Optional overrides for SAM2AutomaticMaskGenerator
                parameters (e.g., ``points_per_side``, ``pred_iou_thresh``,
                ``stability_score_thresh``, ``box_nms_thresh``).
            enable_material_classification: Enable CLIP-based material labeling.
            material_confidence_threshold: Confidence threshold for material labels.
            material_classification_strict: Raise material classification load or
                inference failures instead of leaving masks unlabeled.
            tiling: Optional tiling configuration for large-image segmentation.
                When enabled, processes images in tiles to manage memory.

        Raises:
            ValueError: If ``model_size`` is invalid, ``prefer_hf_pipeline=True``
                without ``repo_id``, or repo-backed loading lacks a pinned
                revision.
            FileNotFoundError: If local checkpoint path does not exist (when
                checkpoint loading is required).
            ImportError: If SAM2 package or required dependencies are missing.
        """
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model_size '{model_size}', " f"must be one of {list(self.MODEL_CONFIGS.keys())}")

        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.model_config = model_config or self.MODEL_CONFIGS[model_size]
        self.expected_sha256 = (
            _validate_sha256_hex(expected_sha256) if expected_sha256 is not None else self.CHECKPOINT_SHA256.get(model_size)
        )
        self.repo_id = repo_id
        self.revision = revision
        self.prefer_hf_pipeline = False if prefer_hf_pipeline is None else bool(prefer_hf_pipeline)
        self.generator_kwargs = dict(generator_kwargs or {})

        if self.prefer_hf_pipeline and not self.repo_id:
            raise ValueError("prefer_hf_pipeline=True requires repo_id")
        if self.prefer_hf_pipeline:
            assert self.repo_id is not None
            try:
                self.revision = resolve_model_lock_revision(
                    self.repo_id,
                    self.revision,
                    strict=True,
                    context="SAM2",
                )
            except ModelLockError as exc:
                raise ValueError(str(exc)) from exc
            if not self.revision:
                raise ValueError("repo_id-based SAM2 loading requires a pinned revision (40-char commit SHA)")

        # Determine checkpoint path
        if checkpoint_path is None and not self.prefer_hf_pipeline:
            checkpoint_path = os.path.join("checkpoints", self.DEFAULT_CHECKPOINTS[model_size])
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

        # Check checkpoint exists (with helpful error message)
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {self.checkpoint_path}\n"
                f"Download from: https://github.com/facebookresearch/sam2\n"
                f"Or use: python scripts/download_sam2_checkpoint.py"
            )

        self._model: Any = None
        self._mask_generator: Any = None
        self._image_predictor: Any = None
        self._video_predictor: Any = None  # Initialized by _segment_video when needed
        self._hf_mask_generator: Any = None
        self._hf_model: Any = None
        self._hf_processor: Any = None
        self._hf_video_checkpoint_path: Optional[Path] = None

        # N-3 per-instance image-digest memo: lifetime-scoped to this
        # backend so concurrent forward() calls on the same image reuse
        # the digest instead of rehashing. Cleared by ``unload()``.
        from transformation_portal.spatial_ai.segmentation._content_digest import ArrayDigestCache

        self._image_digest_cache: ArrayDigestCache = ArrayDigestCache()

        # Material classification (optional)
        self.enable_material_classification = enable_material_classification
        self.material_confidence_threshold = material_confidence_threshold
        self.material_classification_strict = bool(material_classification_strict)
        self._material_classifier: Any = None
        if enable_material_classification:
            from transformation_portal.spatial_ai.segmentation.material_classifier import MaterialClassifier

            self._material_classifier = MaterialClassifier(
                device=self.device,
                confidence_threshold=material_confidence_threshold,
                strict=self.material_classification_strict,
            )
            logger.info("Material classification enabled")

        self.tiling = tiling or SegmentationTilingConfig(enabled=False)
        self.tiled_engine: Optional[TiledSegmentationEngine] = None

        logger.info(
            "SAM2Backend initialized: model=%s device=%s checkpoint=%s repo_id=%s revision=%s "
            "model_config=%s prefer_hf_pipeline=%s material_classification=%s",
            model_size,
            self.device,
            None if self.checkpoint_path is None else self.checkpoint_path.name,
            self.repo_id,
            self.revision,
            self.model_config,
            self.prefer_hf_pipeline,
            enable_material_classification,
        )

    @classmethod
    def _resolve_device(cls, requested_device: str) -> str:
        """Resolve a device request to an available execution device."""
        if requested_device not in cls.SUPPORTED_DEVICES:
            raise ValueError(f"Invalid device '{requested_device}', " f"must be one of {sorted(cls.SUPPORTED_DEVICES)}")

        # CPU always exists; avoid importing torch for the common explicit CPU path.
        if requested_device == "cpu":
            return "cpu"

        try:
            import torch
        except ImportError:
            if requested_device != "cpu":
                logger.warning(
                    "Torch is not installed; falling back from device '%s' to 'cpu'",
                    requested_device,
                )
            return "cpu"

        cuda_available = bool(torch.cuda.is_available())
        mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

        if requested_device == "auto":
            if mps_available:
                return "mps"
            if cuda_available:
                return "cuda"
            return "cpu"

        if requested_device == "cuda" and not cuda_available:
            fallback = "mps" if mps_available else "cpu"
            logger.warning("Requested device 'cuda' unavailable; falling back to '%s'", fallback)
            return fallback

        if requested_device == "mps" and not mps_available:
            fallback = "cuda" if cuda_available else "cpu"
            logger.warning("Requested device 'mps' unavailable; falling back to '%s'", fallback)
            return fallback

        return requested_device

    @staticmethod
    def _validate_checkpoint_sha256(checkpoint_path: Path, expected_sha256: str) -> None:
        """Validate checkpoint bytes against the trusted SHA-256 digest."""
        actual_sha256 = _compute_file_sha256(checkpoint_path)
        if actual_sha256 != expected_sha256:
            raise SAM2CheckpointIntegrityError(
                f"SHA-256 mismatch for SAM2 checkpoint {checkpoint_path}: " f"expected {expected_sha256}, got {actual_sha256}"
            )

    def _load_model(self) -> None:
        """Lazy load SAM2 model and mask generator.

        Raises:
            ImportError: If sam2 package missing.
            RuntimeError: If model loading fails.
        """
        if self._model is not None or self._hf_mask_generator is not None:
            return  # Already loaded

        if self.repo_id and self.prefer_hf_pipeline:
            try:
                self._load_huggingface_path()
                return
            except Exception as exc:
                self.unload_model()
                if self.checkpoint_path is None:
                    raise RuntimeError(
                        f"Failed to load SAM2 from repo_id={self.repo_id} revision={self.revision}: {exc}"
                    ) from exc
                logger.warning(
                    "Falling back from Hugging Face SAM2 load to local checkpoint path due to error: %s",
                    exc,
                )

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError("SAM2 requires sam2 and torch. Install with: pip install sam2 torch torchvision") from e

        # Config name for Hydra (sam2 package initializes config module in __init__.py)
        if self.checkpoint_path is None:
            raise RuntimeError(
                "No SAM2 checkpoint_path is available for the official loader. "
                "Either provide checkpoint_path or enable the pinned Hugging Face path."
            )

        config_name = self.model_config
        logger.info(f"Loading SAM2 model: {config_name} @ {self.checkpoint_path.name}")

        try:
            if self.expected_sha256:
                self._validate_checkpoint_sha256(self.checkpoint_path, self.expected_sha256)
            # Build SAM2 model (uses Hydra config module initialized by sam2 package)
            self._model = build_sam2(
                config_file=config_name,
                ckpt_path=str(self.checkpoint_path),
                device=self.device,
                mode="eval",
            )

            # Create automatic mask generator (for auto mode)
            self._mask_generator = SAM2AutomaticMaskGenerator(
                model=self._model,
                points_per_side=self.generator_kwargs.get("points_per_side", 32),
                points_per_batch=self.generator_kwargs.get("points_per_batch", 64),
                pred_iou_thresh=self.generator_kwargs.get("pred_iou_thresh", 0.88),
                stability_score_thresh=self.generator_kwargs.get("stability_score_thresh", 0.85),
                box_nms_thresh=self.generator_kwargs.get("box_nms_thresh", 0.7),
                crop_n_layers=self.generator_kwargs.get("crop_n_layers", 1),
                crop_nms_thresh=self.generator_kwargs.get("crop_nms_thresh", 0.7),
            )

            # Create image predictor (for prompted mode)
            self._image_predictor = SAM2ImagePredictor(self._model)

            logger.info("SAM2 model loaded successfully")

        except SAM2CheckpointIntegrityError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model: {e}") from e

    def segment(
        self,
        seg_input: SegmentationInput,
    ) -> SegmentationResult:
        """Segment image with SAM2.

        Args:
            seg_input: Validated segmentation input contract.

        Returns:
            SegmentationResult with masks, scores, and metadata.

        Raises:
            ValueError: If input contract violated or mode unsupported.
            RuntimeError: If inference fails.
        """
        # Contract validation already done in SegmentationInput.__post_init__

        # Execute segmentation based on mode
        if seg_input.mode == "video":
            return self._segment_video(seg_input)

        # Lazy load model
        self._load_model()

        if self.tiling.enabled and seg_input.mode in self.tiling.apply_to_modes:
            if self.tiled_engine is None:
                self.tiled_engine = self._build_default_tiled_engine()
            # N-3: thread the upstream cache layer's precomputed digest
            # (when set) so the image is hashed at most once per pipeline
            # run; otherwise the per-instance cache memoizes the first
            # computation for the duration of this backend.
            image_hash = self._stable_image_hash(
                seg_input.image,
                precomputed=getattr(seg_input, "content_digest", None),
            )
            return self.tiled_engine.run(
                backend=self,
                seg_input=seg_input,
                image_hash=image_hash,
                config=self.tiling,
            )

        if seg_input.mode == "auto":
            return self._segment_auto(seg_input)
        if seg_input.mode in ["points", "bbox"]:
            return self._segment_prompted(seg_input)
        raise ValueError(f"Unsupported mode: {seg_input.mode}")

    def global_seed_pass(
        self,
        *,
        image_linear: np.ndarray,
        image_hash: str,
        longest_side: int,
        rng_seed: int,
    ) -> GlobalSeedHints:
        del rng_seed
        H, W = image_linear.shape[:2]
        scale = min(1.0, float(longest_side) / float(max(H, W)))
        low_h = max(1, int(round(H * scale)))
        low_w = max(1, int(round(W * scale)))
        return GlobalSeedHints(
            image_hash=image_hash,
            low_res_longest_side=longest_side,
            low_res_W=low_w,
            low_res_H=low_h,
            scale_x=low_w / max(1.0, float(W)),
            scale_y=low_h / max(1.0, float(H)),
            meta={"backend": self.model_size},
        )

    def segment_tile(
        self,
        *,
        tile_linear: np.ndarray,
        image_hash: str,
        tile_spec: TileSpec,
        mode: str,
        prompts: Optional[Dict],
        global_hints: Optional[GlobalSeedHints],
        rng_seed: int,
    ) -> tuple[TileInstance, ...]:
        """Segment a single tile using existing image-mode SAM2 inference.

        This default implementation intentionally ignores ``global_hints`` and
        ``rng_seed`` and delegates to current monolithic SAM2 methods for each
        tile. The parameters remain in the interface for future quality parity
        enhancements where global context and deterministic sampling are used.
        """
        del image_hash, global_hints, rng_seed
        tile_prompts = self._translate_prompts_to_tile(prompts, tile_spec, mode)
        seg_input = SegmentationInput(
            image=tile_linear,
            gamma=1.0,
            mode=cast(Literal["auto", "points", "bbox", "video"], mode),
            prompts=tile_prompts,
        )
        seg_result = self._segment_auto(seg_input) if mode == "auto" else self._segment_prompted(seg_input)

        instances = []
        for idx in range(seg_result.masks.shape[0]):
            mask = seg_result.masks[idx].astype(np.float32, copy=False)
            instances.append(
                TileInstance(
                    local_id=f"{tile_spec.tile_id}:{idx}",
                    score=float(np.clip(seg_result.scores[idx], 0.0, 1.0)),
                    stability_score=float(seg_result.metadata[idx].stability_score),
                    soft_mask=SoftMaskPatch(
                        bbox=BBox(0, 0, int(mask.shape[1]), int(mask.shape[0])),
                        values=mask,
                        space="prob",
                    ),
                    material_label=seg_result.metadata[idx].material_label,
                    material_confidence=seg_result.metadata[idx].material_confidence,
                )
            )
        return tuple(instances)

    def _translate_prompts_to_tile(self, prompts: Optional[Dict], tile_spec: TileSpec, mode: str) -> Optional[Dict]:
        if not prompts or mode == "auto":
            return prompts
        translated = dict(prompts)
        offset_x, offset_y = tile_spec.bbox.x0, tile_spec.bbox.y0
        tile_width = max(0, tile_spec.bbox.x1 - tile_spec.bbox.x0)
        tile_height = max(0, tile_spec.bbox.y1 - tile_spec.bbox.y0)

        if mode == "points" and "points" in translated:
            points = []
            for point in translated["points"]:
                x = point[0] - offset_x
                y = point[1] - offset_y
                if tile_width > 0:
                    x = max(0.0, min(float(x), float(tile_width - 1)))
                if tile_height > 0:
                    y = max(0.0, min(float(y), float(tile_height - 1)))
                points.append([x, y])
            translated["points"] = points
            return translated
        if mode == "bbox" and "bbox" in translated:
            x0, y0, x1, y1 = translated["bbox"]
            x0 -= offset_x
            y0 -= offset_y
            x1 -= offset_x
            y1 -= offset_y
            if tile_width > 0:
                x0 = max(0.0, min(float(x0), float(tile_width)))
                x1 = max(0.0, min(float(x1), float(tile_width)))
            if tile_height > 0:
                y0 = max(0.0, min(float(y0), float(tile_height)))
                y1 = max(0.0, min(float(y1), float(tile_height)))
            translated["bbox"] = [x0, y0, x1, y1]
            return translated
        return translated

    def _build_default_tiled_engine(self) -> TiledSegmentationEngine:
        return TiledSegmentationEngine(
            planner=UniformTilingPlanner(),
            merger=BinaryUnionTileMerger(),
            validator=SeamMergeValidator(),
        )

    def _stable_image_hash(
        self,
        image: Optional[np.ndarray],
        *,
        precomputed: Optional[str] = None,
    ) -> str:
        """Compute (or reuse) a deterministic SHA-256 hash for an image.

        Delegates to the per-instance ``_image_digest_cache``, which is
        backed by ``_content_digest.ArrayDigestCache``. Repeat calls with
        the same array object hit the cache in O(1); the first call
        computes the full digest via the shared
        ``compute_array_sha256`` helper (same formula previously
        duplicated here).

        If ``precomputed`` is provided (e.g. threaded down from
        ``SegmentationInput.content_digest`` where the upstream cache
        layer already hashed the image), the cache adopts it instead of
        recomputing. Output is unchanged versus the legacy
        implementation: shape repr + dtype repr + raw uint8 buffer view.
        """
        return self._image_digest_cache.get_or_compute(image, override=precomputed)

    def unload(self) -> None:
        """Release loaded model/material references and best-effort device cache."""
        self.unload_model()

    def unload_model(self) -> None:
        """Release loaded model/material references and best-effort device cache."""
        self._model = None
        self._mask_generator = None
        self._image_predictor = None
        self._video_predictor = None
        self._hf_mask_generator = None
        self._hf_model = None
        self._hf_processor = None
        self._material_classifier = None
        # N-3: release any cached image digests with the model.
        self._image_digest_cache.clear()
        try:
            import torch
        except (ImportError, OSError):
            # ImportError: torch not installed
            # OSError: torch installed but native libraries missing/mislinked
            return
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clone_for_device(self, device: str) -> "SAM2Backend":
        """Create an equivalent backend bound to a new execution device."""
        return SAM2Backend(
            model_size=self.model_size,
            device=cast(Literal["auto", "cuda", "cpu", "mps"], device),
            checkpoint_path=None if self.checkpoint_path is None else str(self.checkpoint_path),
            model_config=self.model_config,
            expected_sha256=self.expected_sha256,
            repo_id=self.repo_id,
            revision=self.revision,
            prefer_hf_pipeline=self.prefer_hf_pipeline,
            generator_kwargs=self.generator_kwargs,
            enable_material_classification=self.enable_material_classification,
            material_confidence_threshold=self.material_confidence_threshold,
            material_classification_strict=self.material_classification_strict,
            tiling=self.tiling,
        )

    @staticmethod
    def _to_numpy_array(value: Any, *, dtype: Any = None) -> np.ndarray:
        """Convert SAM2-style outputs to NumPy arrays without assuming tensor type."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    @staticmethod
    def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
        """Normalize arrays to uint8 RGB for SAM2 backends."""
        if image.dtype in (np.float32, np.float64):
            return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        if image.dtype != np.uint8:
            return np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _make_mask_metadata(
        self,
        *,
        area: int,
        bbox: tuple[int, int, int, int],
        stability_score: float,
        material_label: Optional[str] = None,
        material_confidence: Optional[float] = None,
        is_empty: bool = False,
    ) -> MaskMetadata:
        """Build MaskMetadata while tolerating future field changes."""
        return make_mask_metadata(
            area=area,
            bbox=bbox,
            stability_score=stability_score,
            material_label=material_label,
            material_confidence=material_confidence,
            is_empty=is_empty,
        )

    def _apply_material_labels(
        self,
        image_uint8: np.ndarray,
        masks: np.ndarray,
        metadata_list: list[MaskMetadata],
    ) -> None:
        """Populate optional material labels when classifier support is enabled."""
        if self.enable_material_classification and self._material_classifier is not None:
            try:
                if not self._material_classifier.is_available():
                    if self.material_classification_strict:
                        raise RuntimeError(
                            "Material classification is enabled in strict mode, but the classifier is unavailable."
                        )
                    return
                logger.info("Running material classification...")
                material_results = self._material_classifier.classify_masks(image_uint8, masks)
            except Exception as exc:
                if self.material_classification_strict:
                    raise
                logger.warning("Material classification failed; leaving SAM2 masks unlabeled: %s", exc)
                return

            for idx, (label, confidence) in enumerate(material_results):
                if idx >= len(metadata_list):
                    break
                metadata_list[idx].material_label = label
                metadata_list[idx].material_confidence = confidence

    def _load_huggingface_path(self) -> None:
        """Load SAM2 via pinned Hugging Face repo-backed surfaces."""
        if not self.repo_id or not self.revision:
            raise RuntimeError("Hugging Face SAM2 loading requires pinned repo_id and revision")

        from transformers import Sam2Model, Sam2Processor, pipeline

        pipeline_device: Any
        if self.device == "cuda":
            pipeline_device = 0
        elif self.device == "cpu":
            pipeline_device = -1
        else:
            pipeline_device = self.device

        hf_mask_generator = pipeline(
            "mask-generation",
            model=self.repo_id,
            revision=self.revision,
            device=pipeline_device,
        )
        hf_model = getattr(hf_mask_generator, "model", None)
        hf_processor = getattr(hf_mask_generator, "image_processor", None)
        if hf_processor is None:
            hf_processor = getattr(hf_mask_generator, "processor", None)

        if hf_model is None:
            hf_model = Sam2Model.from_pretrained(self.repo_id, revision=self.revision)
        target_device = self.device if self.device in {"cuda", "mps"} else "cpu"
        hf_model = cast(Any, hf_model).to(target_device)
        cast(Any, hf_model).eval()
        if hf_processor is None or not hasattr(hf_processor, "post_process_masks"):
            hf_processor = Sam2Processor.from_pretrained(self.repo_id, revision=self.revision)

        self._hf_mask_generator = hf_mask_generator
        self._hf_model = hf_model
        self._hf_processor = hf_processor

    @staticmethod
    def _hf_offline_mode_enabled() -> bool:
        """Return True when HuggingFace/Transformers offline flags are enabled."""
        return os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

    def _iter_hf_checkpoint_candidates(self) -> tuple[str, ...]:
        """Return plausible SAM2 checkpoint filenames for pinned repo-backed loads."""
        default_name = self.DEFAULT_CHECKPOINTS[self.model_size]
        candidates = [default_name]
        if default_name.startswith("sam2_"):
            candidates.append(default_name.replace("sam2_", "sam2.1_", 1))
        if default_name.startswith("sam2.1_"):
            candidates.append(default_name.replace("sam2.1_", "sam2_", 1))
        return tuple(dict.fromkeys(candidates))

    def _resolve_hf_video_checkpoint_path(self) -> Path:
        """Resolve a local checkpoint file for the official SAM2 video predictor path."""
        if self._hf_video_checkpoint_path is not None and self._hf_video_checkpoint_path.is_file():
            return self._hf_video_checkpoint_path
        if not self.repo_id or not self.revision:
            raise RuntimeError("Pinned repo_id and revision are required for SAM2 video checkpoint resolution")

        try:
            from huggingface_hub import hf_hub_download, try_to_load_from_cache
        except ImportError as exc:
            raise RuntimeError(
                "repo_id-based SAM2 video tracking requires huggingface_hub to resolve the pinned checkpoint"
            ) from exc

        candidates = self._iter_hf_checkpoint_candidates()
        failures: list[str] = []
        offline = self._hf_offline_mode_enabled()

        for filename in candidates:
            cached_path: Any = None
            try:
                cached_path = try_to_load_from_cache(repo_id=self.repo_id, filename=filename, revision=self.revision)
            except TypeError:
                cached_path = try_to_load_from_cache(repo_id=self.repo_id, filename=filename)
            except (OSError, ValueError) as exc:
                failures.append(f"{filename}: cache lookup failed ({exc})")
                cached_path = None

            if isinstance(cached_path, str) and Path(cached_path).is_file():
                self._hf_video_checkpoint_path = Path(cached_path)
                return self._hf_video_checkpoint_path

            try:
                resolved = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    revision=self.revision,
                    local_files_only=offline,
                )
            except Exception as exc:  # pragma: no cover - exercised via focused unit stubs
                failures.append(f"{filename}: {exc}")
                continue

            resolved_path = Path(resolved)
            if resolved_path.is_file():
                self._hf_video_checkpoint_path = resolved_path
                return self._hf_video_checkpoint_path

        failure_summary = "; ".join(failures) if failures else "no matching checkpoint candidates found"
        raise RuntimeError(
            f"Unable to resolve SAM2 video checkpoint for repo_id={self.repo_id} revision={self.revision}. "
            f"Tried {', '.join(candidates)}. {failure_summary}"
        )

    def _extract_sam2_predictions(self, output: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract masks plus IoU/stability scores from a SAM2 output object.

        ``pred_masks`` is required. Missing or ``None`` confidence attributes
        fall back to ``1.0`` so older stub outputs and partial mocks still
        satisfy the segmentation contract.
        """
        raw_masks = getattr(output, "pred_masks", None)
        if raw_masks is None:
            raise AttributeError("SAM2 output missing required pred_masks attribute")
        masks = self._to_numpy_array(raw_masks, dtype=bool)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]

        count = masks.shape[0]

        def _extract_scores(attr_name: str) -> np.ndarray:
            raw_value = getattr(output, attr_name, None)
            if raw_value is None:
                return np.ones(count, dtype=np.float32)
            scores = self._to_numpy_array(raw_value, dtype=np.float32).reshape(-1)
            if scores.shape != (count,):
                return np.ones(count, dtype=np.float32)
            return np.clip(scores, 0.0, 1.0)

        iou_scores = _extract_scores("iou_predictions")
        stability_scores = _extract_scores("stability_scores")
        return masks, iou_scores, stability_scores

    def _segment_auto(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Automatic mask generation (entire image).

        Uses SAM2's automatic mask generator to detect all objects
        in the image without any prompts.

        Args:
            seg_input: Validated segmentation input.

        Returns:
            SegmentationResult with all detected masks.

        Raises:
            RuntimeError: If mask generation fails.
        """
        image = seg_input.image
        assert image is not None
        image_uint8 = self._ensure_uint8_rgb(image)

        if self._hf_mask_generator is not None:
            try:
                pil_image = Image.fromarray(image_uint8, mode="RGB")
                outputs = self._hf_mask_generator(pil_image, **self.generator_kwargs)

                raw_masks = outputs.get("masks", [])
                raw_scores = outputs.get("scores")

                if not raw_masks:
                    logger.warning("SAM2 HF pipeline found no masks in auto mode")
                    return SegmentationResult(
                        masks=np.zeros((0, *image_uint8.shape[:2]), dtype=bool),
                        scores=np.zeros((0,), dtype=np.float32),
                        metadata=[],
                    )

                mask_arrays: list[np.ndarray] = []
                for raw_mask in raw_masks:
                    arr = self._to_numpy_array(raw_mask)
                    arr = np.squeeze(arr)
                    if arr.ndim != 2:
                        raise RuntimeError(f"Unexpected HF SAM2 auto-mask shape: {arr.shape}")
                    mask_arrays.append(arr.astype(bool, copy=False))

                masks = np.stack(mask_arrays, axis=0)
                if raw_scores is not None:
                    scores = self._to_numpy_array(raw_scores, dtype=np.float32).reshape(-1)
                else:
                    scores = np.ones((masks.shape[0],), dtype=np.float32)
                if scores.shape != (masks.shape[0],):
                    scores = np.ones((masks.shape[0],), dtype=np.float32)

                order = np.argsort(-scores, kind="stable")
                masks = masks[order]
                scores = np.clip(scores[order], 0.0, 1.0)

                metadata_list = []
                for idx, mask in enumerate(masks):
                    metadata_list.append(metadata_from_mask(mask, stability_score=float(scores[idx])))

                self._apply_material_labels(image_uint8, masks, metadata_list)
                return SegmentationResult(masks=masks, scores=scores, metadata=metadata_list)
            except Exception as e:
                raise RuntimeError(f"SAM2 HF auto mode segmentation failed: {e}") from e

        try:
            # Generate masks
            masks_data: list[dict[str, Any]] = self._mask_generator.generate(image_uint8)

            # Extract masks and scores
            if not masks_data:
                # No masks found - return empty result
                logger.warning("SAM2 found no masks in auto mode")
                return SegmentationResult(
                    masks=np.zeros((0, *image.shape[:2]), dtype=bool),
                    scores=np.zeros(0, dtype=np.float32),
                    metadata=[],  # Empty list for empty result
                )

            # Convert SAM2 output to our format
            masks = np.stack([m["segmentation"] for m in masks_data])
            iou_scores = np.array([m["predicted_iou"] for m in masks_data], dtype=np.float32)
            stability_scores = np.array([m["stability_score"] for m in masks_data], dtype=np.float32)

            # Use average of IoU and stability as final score
            scores = (iou_scores + stability_scores) / 2.0

            # Create metadata for each mask
            metadata_list = []
            for m, stab_score in zip(masks_data, stability_scores):
                bbox_raw = m["bbox"]  # SAM2 format: [x, y, w, h]
                # SAM2 returns bbox in [x, y, width, height] format already
                bbox_xywh = (
                    int(bbox_raw[0]),
                    int(bbox_raw[1]),
                    int(bbox_raw[2]),
                    int(bbox_raw[3]),
                )
                metadata_list.append(
                    self._make_mask_metadata(area=int(m["area"]), bbox=bbox_xywh, stability_score=float(stab_score))
                )

            self._apply_material_labels(image_uint8, masks, metadata_list)

            logger.info(f"SAM2 auto mode: generated {len(masks)} masks")

            return SegmentationResult(
                masks=masks,
                scores=scores,
                metadata=metadata_list,
            )

        except Exception as e:
            raise RuntimeError(f"SAM2 auto mode segmentation failed: {e}") from e

    def _segment_prompted(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Prompted segmentation (points or bounding boxes).

        Args:
            seg_input: Validated segmentation input with prompts.

        Returns:
            SegmentationResult with prompted masks.

        Raises:
            ValueError: If prompts are invalid.
            RuntimeError: If segmentation fails.
        """
        image = seg_input.image
        assert image is not None
        mode = seg_input.mode
        image_uint8 = self._ensure_uint8_rgb(image)

        if self._hf_model is not None and self._hf_processor is not None:
            try:
                import torch

                pil_image = Image.fromarray(image_uint8, mode="RGB")

                if mode == "points":
                    if seg_input.prompts is None or "points" not in seg_input.prompts:
                        raise ValueError("Points mode requires 'points' in prompts dict")
                    raw_points = seg_input.prompts["points"]
                    raw_labels = seg_input.prompts.get("labels", [1] * len(raw_points))
                    point_batch = [[[float(x), float(y)] for x, y in raw_points]]
                    label_batch = [[int(v) for v in raw_labels]]
                    inputs = self._hf_processor(
                        images=pil_image,
                        input_points=[point_batch],
                        input_labels=[label_batch],
                        return_tensors="pt",
                    )
                elif mode == "bbox":
                    if seg_input.prompts is None or "bbox" not in seg_input.prompts:
                        raise ValueError("Bbox mode requires 'bbox' in prompts dict")
                    bbox = [float(v) for v in seg_input.prompts["bbox"]]
                    inputs = self._hf_processor(
                        images=pil_image,
                        input_boxes=[[bbox]],
                        return_tensors="pt",
                    )
                else:
                    raise ValueError(f"Unsupported prompted mode: {mode}")

                inputs = {
                    key: value.to(self._hf_model.device) if hasattr(value, "to") else value for key, value in inputs.items()
                }
                with torch.no_grad():
                    outputs = self._hf_model(**inputs, multimask_output=True)

                post_masks = self._hf_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"],
                )[0]
                masks = self._to_numpy_array(post_masks)
                if masks.ndim == 4:
                    masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
                elif masks.ndim != 3:
                    raise RuntimeError(f"Unexpected HF prompted mask shape: {masks.shape}")
                masks = masks > 0

                scores = self._to_numpy_array(outputs.iou_scores, dtype=np.float32).reshape(-1)
                if scores.shape != (masks.shape[0],):
                    scores = np.ones((masks.shape[0],), dtype=np.float32)

                order = np.argsort(-scores, kind="stable")
                masks = masks[order]
                scores = np.clip(scores[order], 0.0, 1.0)

                metadata_list = []
                for idx, mask in enumerate(masks):
                    metadata_list.append(metadata_from_mask(mask, stability_score=float(scores[idx])))

                self._apply_material_labels(image_uint8, masks, metadata_list)
                return SegmentationResult(masks=masks, scores=scores, metadata=metadata_list)
            except Exception as e:
                raise RuntimeError(f"SAM2 HF {mode} mode segmentation failed: {e}") from e

        try:
            # Set image in predictor
            self._image_predictor.set_image(image_uint8)

            if mode == "points":
                # Extract points from prompts dict
                if seg_input.prompts is None or "points" not in seg_input.prompts:
                    raise ValueError("Points mode requires 'points' in prompts dict")

                points = np.array(seg_input.prompts["points"])
                labels = np.array(seg_input.prompts.get("labels", [1] * len(points)))  # Default to foreground

                # Predict masks
                prediction = self._image_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,  # Get multiple mask proposals
                )

            elif mode == "bbox":
                # Extract bbox from prompts dict
                if seg_input.prompts is None or "bbox" not in seg_input.prompts:
                    raise ValueError("Bbox mode requires 'bbox' in prompts dict")

                bbox_prompt = np.array(seg_input.prompts["bbox"])  # [x1, y1, x2, y2]

                # Predict masks
                prediction = self._image_predictor.predict(
                    box=bbox_prompt,
                    multimask_output=True,
                )

            else:
                raise ValueError(f"Unsupported prompted mode: {mode}")

            if isinstance(prediction, tuple):
                if len(prediction) < 2:
                    raise RuntimeError("SAM2 predictor returned an unexpected prompted output tuple")
                prediction = SimpleNamespace(
                    pred_masks=prediction[0],
                    iou_predictions=prediction[1],
                    stability_scores=prediction[1],
                )

            masks, scores, stability_scores = self._extract_sam2_predictions(prediction)

            # SAM2ImagePredictor may return proposals in arbitrary order.
            # Normalize to a deterministic highest-confidence-first order so
            # prompted mode has a stable primary mask across environments.
            scores = np.asarray(scores, dtype=np.float32)

            # Convert masks to correct format
            if masks.ndim == 3:
                # Already (N, H, W)
                pass
            elif masks.ndim == 2:
                # Single mask (H, W) - expand to (1, H, W)
                masks = masks[np.newaxis, ...]

            if masks.shape[0] > 1:
                areas = masks.reshape(masks.shape[0], -1).sum(axis=1, dtype=np.int64)
                ordered_indices = [
                    idx
                    for idx, _ in sorted(
                        enumerate(zip(scores, areas)),
                        key=lambda item: (float(item[1][0]), int(item[1][1])),
                        reverse=True,
                    )
                ]
                masks = masks[ordered_indices]
                scores = scores[ordered_indices]
                stability_scores = stability_scores[ordered_indices]

            # Create metadata for each mask
            metadata_list = []
            valid_indices: list[int] = []
            for i, mask in enumerate(masks):
                area = int(mask.sum())
                if area <= 0:
                    continue

                metadata_list.append(metadata_from_mask(mask, stability_score=float(stability_scores[i])))
                valid_indices.append(i)

            if not metadata_list:
                logger.info("SAM2 %s mode produced only zero-area masks; returning empty result", mode)
                return SegmentationResult(
                    masks=np.zeros((0, *image_uint8.shape[:2]), dtype=bool),
                    scores=np.zeros((0,), dtype=np.float32),
                    metadata=[],
                )

            masks = masks[valid_indices]
            scores = scores[valid_indices]
            self._apply_material_labels(image_uint8, masks, metadata_list)

            logger.info(f"SAM2 {mode} mode: generated {len(masks)} masks")

            return SegmentationResult(
                masks=masks,
                scores=scores.astype(np.float32),
                metadata=metadata_list,
            )

        except Exception as e:
            raise RuntimeError(f"SAM2 {mode} mode segmentation failed: {e}") from e

    def _segment_video(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Video segmentation with temporal tracking (Phase 4A).

        Uses SAM2VideoPredictor to track objects across video frames.
        Requires:
        - video_path: Path to video file (MP4/MOV)
        - prompts: Initial frame prompts (points or bbox)

        Args:
            seg_input: Validated segmentation input with video_path.

        Returns:
            SegmentationResult with masks for all tracked frames.

        Raises:
            RuntimeError: If video tracking fails.
            ImportError: If SAM2 video components missing.
        """
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e:
            raise ImportError("SAM2 video predictor not available. Install sam2 package: pip install sam2") from e

        # Validate video file exists
        assert seg_input.video_path is not None
        video_path = Path(seg_input.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Build video predictor (lazy load)
        if not hasattr(self, "_video_predictor") or self._video_predictor is None:
            logger.info(f"Loading SAM2 video predictor: {self.model_size} on {self.device}")
            config_name = self.model_config
            checkpoint_path = (
                self._resolve_hf_video_checkpoint_path()
                if self.prefer_hf_pipeline and self.checkpoint_path is None
                else self.checkpoint_path
            )
            if checkpoint_path is None:
                raise RuntimeError(
                    "SAM2 video tracking requires either a trusted checkpoint_path or a pinned repo_id/revision"
                )
            if self.expected_sha256:
                self._validate_checkpoint_sha256(Path(checkpoint_path), self.expected_sha256)

            self._video_predictor = build_sam2_video_predictor(
                config_file=config_name,
                ckpt_path=str(checkpoint_path),
                device=self.device,
            )

        # Initialize inference state
        logger.info(f"Initializing video state: {video_path}")
        inference_state = self._video_predictor.init_state(
            video_path=str(video_path),
            offload_video_to_cpu=False,  # Keep on GPU for speed
            offload_state_to_cpu=False,
        )

        # Extract prompt information
        prompts = seg_input.prompts
        assert prompts is not None
        frame_idx = prompts.get("frame_idx", 0)
        object_id = prompts.get("object_id", 1)

        # Add prompts to initial frame
        if "points" in prompts:
            points = np.array(prompts["points"])  # [[x, y], ...]
            labels = np.array(prompts.get("labels", [1] * len(points)))  # Default: foreground

            logger.info(f"Adding {len(points)} point prompts at frame {frame_idx}, object {object_id}")
            _, out_obj_ids, out_mask_logits = self._video_predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )

        elif "bbox" in prompts:
            bbox_prompt = np.array(prompts["bbox"])  # [x1, y1, x2, y2]
            logger.info(f"Adding bbox prompt at frame {frame_idx}, object {object_id}")

            # Prefer native SAM2 bbox API when available.
            add_points_or_box = getattr(self._video_predictor, "add_new_points_or_box", None)
            if callable(add_points_or_box):
                add_points_or_box = cast(Callable[..., Any], add_points_or_box)
                _, out_obj_ids, out_mask_logits = add_points_or_box(  # pylint: disable=not-callable
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=object_id,
                    box=bbox_prompt,
                    clear_old_points=True,
                )
            else:
                # Backward compatibility for older predictor API:
                # bbox corners are represented with SAM2-special labels 2 and 3.
                points = np.array([[bbox_prompt[0], bbox_prompt[1]], [bbox_prompt[2], bbox_prompt[3]]])
                labels = np.array([2, 3], dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self._video_predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )

        else:
            raise ValueError("Video mode requires 'points' or 'bbox' in prompts")

        # Propagate prompts across all video frames
        logger.info("Propagating masks across video frames...")
        video_segments = {}  # {frame_idx: {obj_id: mask}}

        for out_frame_idx, out_obj_ids, out_mask_logits in self._video_predictor.propagate_in_video(inference_state):
            frame_segments = {}
            for i, obj_id in enumerate(out_obj_ids):
                mask_logits = out_mask_logits[i]
                if hasattr(mask_logits, "detach"):
                    frame_segments[obj_id] = (mask_logits > 0.0).detach().cpu().numpy()
                else:
                    frame_segments[obj_id] = np.asarray(mask_logits) > 0.0
            video_segments[out_frame_idx] = frame_segments

        # Extract masks for tracked object
        masks = []
        scores = []
        metadata_list = []
        num_frames = inference_state["num_frames"]

        for frame_idx in range(num_frames):
            if frame_idx in video_segments and object_id in video_segments[frame_idx]:
                mask_data = video_segments[frame_idx][object_id]

                mask = np.asarray(mask_data).squeeze().astype(bool)

                masks.append(mask)
                scores.append(1.0)  # Video tracking doesn't provide scores

                metadata_list.append(metadata_from_mask(mask, stability_score=1.0))
            else:
                # Object not found in this frame
                logger.warning(f"Object {object_id} not found in frame {frame_idx}")
                # Create empty mask
                h, w = inference_state["video_height"], inference_state["video_width"]
                empty_mask = np.zeros((h, w), dtype=bool)
                masks.append(empty_mask)
                scores.append(0.0)
                metadata_list.append(
                    make_mask_metadata(
                        area=0,
                        bbox=(0, 0, 1, 1),
                        stability_score=0.0,
                        is_empty=True,
                    )
                )

        logger.info(f"Video tracking complete: {len(masks)} frames, object {object_id}")

        # Clean up state
        self._video_predictor.reset_state(inference_state)

        # Stack masks into (N, H, W) array
        masks_array = np.stack(masks, axis=0)  # (N, H, W)

        return SegmentationResult(
            masks=masks_array,
            scores=np.array(scores),
            metadata=metadata_list,
            temporal_ids=np.full(len(masks), object_id, dtype=int),  # Same ID for all frames
        )


# Utility function for download script
def download_sam2_checkpoint(
    model_size: Literal["base", "large"] = "large",
    output_dir: str = "checkpoints",
    expected_sha256: Optional[str] = None,
) -> Path:
    """Download SAM2 checkpoint from official repository.

    Args:
        model_size: Model variant to download.
        output_dir: Directory to save checkpoint.
        expected_sha256: Optional checksum override. If omitted, uses built-in checksum registry.

    Returns:
        Path to downloaded checkpoint.

    Raises:
        RuntimeError: If download fails.
    """
    import http.client
    from urllib.parse import urlparse

    url = SAM2Backend.CHECKPOINT_URLS[model_size]
    filename = SAM2Backend.DEFAULT_CHECKPOINTS[model_size]
    output_path = Path(output_dir) / filename
    expected = expected_sha256 or SAM2Backend.CHECKPOINT_SHA256.get(model_size)
    if not expected:
        raise RuntimeError(
            f"No expected SHA256 configured for SAM2 {model_size} checkpoint. Provide expected_sha256 explicitly."
        )
    expected = _validate_sha256_hex(expected)
    allowed_hosts = {"dl.fbaipublicfiles.com"}

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_sha = _compute_file_sha256(output_path)
        if existing_sha == expected:
            logger.info(f"Checkpoint already exists and checksum verified: {output_path}")
            return output_path
        logger.warning(
            "Existing checkpoint checksum mismatch for %s. Expected %s, got %s. Re-downloading.",
            output_path,
            expected,
            existing_sha,
        )
        output_path.unlink(missing_ok=True)

    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname not in allowed_hosts:
        raise RuntimeError(f"Refusing to download checkpoint from untrusted URL: {url}")

    logger.info(f"Downloading SAM2 {model_size} checkpoint from {url}...")
    logger.info("This may take several minutes (checkpoint is ~200-400 MB)...")

    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    current_url = url
    max_redirects = 3
    try:
        for _ in range(max_redirects + 1):
            parsed_url = urlparse(current_url)
            host = parsed_url.hostname
            if parsed_url.scheme != "https" or host not in allowed_hosts:
                raise RuntimeError(f"Refusing to download checkpoint from untrusted URL: {current_url}")

            request_path = parsed_url.path or "/"
            if parsed_url.query:
                request_path = f"{request_path}?{parsed_url.query}"

            connection = http.client.HTTPSConnection(host, parsed_url.port or 443, timeout=300)
            try:
                connection.request("GET", request_path, headers={"User-Agent": "transformation-portal/ci"})
                response = connection.getresponse()
                if response.status in {301, 302, 303, 307, 308}:
                    redirect_location = response.getheader("Location")
                    response.read()  # Drain body before reusing loop
                    if not redirect_location:
                        raise RuntimeError("Checkpoint download redirect missing Location header")
                    if redirect_location.startswith("/"):
                        current_url = f"https://{host}{redirect_location}"
                    else:
                        current_url = redirect_location
                    continue
                if response.status != 200:
                    raise RuntimeError(f"Checkpoint download failed with HTTP {response.status}: {response.reason}")

                with temp_path.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)

                actual_sha = _compute_file_sha256(temp_path)
                if actual_sha != expected:
                    raise RuntimeError(
                        f"Checkpoint checksum mismatch for {filename}. " f"Expected {expected}, got {actual_sha}"
                    )

                temp_path.replace(output_path)
                logger.info(f"✅ Downloaded: {output_path}")
                return output_path
            finally:
                connection.close()

        raise RuntimeError("Too many redirects while downloading SAM2 checkpoint")
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download SAM2 checkpoint: {e}") from e
