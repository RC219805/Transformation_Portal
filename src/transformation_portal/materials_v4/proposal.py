"""Strict local SAM2.1 geometric proposals for experimental Materials V4 inference."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .contracts import MaterialsError

SAM2_LARGE_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise MaterialsError("Material inference cancelled")


@dataclass(frozen=True)
class Proposal:
    mask: np.ndarray
    geometric_quality: float
    stability: float


def _validated_counts(record: Any, shape: tuple[int, int]) -> list[int]:
    if not isinstance(record, dict) or set(record) != {"size", "counts"} or tuple(record["size"]) != shape:
        raise MaterialsError("SAM2 returned an invalid proposal RLE geometry")
    counts = record["counts"]
    pixels = math.prod(shape)
    if (
        not isinstance(counts, list)
        or len(counts) > pixels + 1
        or any(type(count) is not int or count < 0 for count in counts)
        or sum(counts) != pixels
    ):
        raise MaterialsError("SAM2 returned unbounded or inconsistent RLE counts")
    return counts


class _CandidateBudget:
    """Conservative candidate accounting, excluding model/allocator overhead.

    SAM2.1 Large decodes at 1024 and retains 256-square low-resolution logits.
    Allow 16 bytes per decoder pixel for float copies and boolean scratch, plus
    128 bytes per proxy pixel for RLE change indices, Python ints/list copies,
    and dense mask intermediates. Retained RLE estimates include upstream
    deepcopy/concatenation headroom. These are admission estimates, not RSS caps.
    """

    def __init__(self, shape: tuple[int, int], config: Any) -> None:
        self.shape = shape
        self.limit = config.max_candidate_bytes
        self.max_candidates = config.max_proposals
        self.count = 0
        self.retained_bytes = 0
        self.per_candidate_scratch = 16 * 1024**2 + 128 * math.prod(shape) + 8 * 256**2 + 4096

    def before_batch(self, points: int) -> None:
        if type(points) is not int or points < 1:
            raise MaterialsError("SAM2 returned an invalid point batch")
        # Official multimask output has three candidates per prompt point.
        if self.retained_bytes + 3 * points * self.per_candidate_scratch > self.limit:
            raise MaterialsError(
                "SAM2 candidate scratch estimate exceeds max_candidate_bytes before batch; "
                "reduce proxy_longest_side or points_per_batch"
            )

    def admit_batch(self, records: Any) -> None:
        if not isinstance(records, list) or self.count + len(records) > self.max_candidates:
            raise MaterialsError("SAM2 pre-NMS candidate count exceeds max_proposals before accumulation")
        added_bytes = sum(128 * len(_validated_counts(record, self.shape)) + 8 * 256**2 + 4096 for record in records)
        if self.retained_bytes + added_bytes > self.limit:
            raise MaterialsError("SAM2 candidate RLE bytes exceed max_candidate_bytes before accumulation")
        self.retained_bytes += added_bytes
        self.count += len(records)


def _decode_rle(record: Any, shape: tuple[int, int]) -> np.ndarray:
    """Decode only bounded, exact SAM uncompressed column-major run lengths."""
    counts = _validated_counts(record, shape)
    pixels = math.prod(shape)
    flat = np.zeros(pixels, dtype=bool)
    cursor = 0
    for index, count in enumerate(counts):
        if index % 2:
            flat[cursor : cursor + count] = True
        cursor += count
    return flat.reshape(shape[1], shape[0]).T.copy()


def generate_proposals(
    image: np.ndarray,
    config: Any,
    *,
    master_shape: tuple[int, int],
    cancelled: Callable[[], bool] | None = None,
) -> tuple[Proposal, ...]:
    """Bound candidate batches before generation and output masks before decoding.

    The model takes an encoded RGB proxy directly; no linear-light claim is made
    about its uint8 input. The official generator's point batches are cancellable.
    """
    check_cancelled(cancelled)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise MaterialsError("SAM2 requires an encoded uint8 RGB proxy")
    budget = _CandidateBudget(image.shape[:2], config)
    budget.before_batch(min(config.points_per_batch, config.points_per_side**2))
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        class BoundedGenerator(SAM2AutomaticMaskGenerator):
            def _process_batch(self, *args: Any, **kwargs: Any) -> Any:
                check_cancelled(cancelled)
                points = args[0] if args else kwargs["points"]
                budget.before_batch(len(points))
                result = super()._process_batch(*args, **kwargs)
                check_cancelled(cancelled)
                budget.admit_batch(result["rles"])
                return result

        model = build_sam2(SAM2_MODEL_CONFIG, ckpt_path=str(config.sam2_checkpoint), device=config.device, mode="eval")
        if getattr(model, "image_size", None) != 1024:
            raise MaterialsError("SAM2 decoder geometry differs from the bounded 1024 configuration")
        if str(next(model.parameters()).device).split(":", 1)[0] != config.device:
            raise MaterialsError("SAM2 actual device differs from the frozen explicit device")
        generator = BoundedGenerator(
            model=model,
            points_per_side=config.points_per_side,
            points_per_batch=config.points_per_batch,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=config.crop_n_layers,
            output_mode="uncompressed_rle",
            min_mask_region_area=0,
            multimask_output=True,
        )
        with torch.inference_mode():
            records = generator.generate(np.array(image, copy=True))
        del generator, model
    except MaterialsError:
        raise
    except Exception as exc:
        raise MaterialsError(f"SAM2 proposal inference failed without fallback: {exc}") from exc
    check_cancelled(cancelled)
    if (
        len(records) > config.max_proposals
        or len(records) * math.prod(master_shape) * 4 > config.max_mask_bytes
        or len(records) * math.prod(image.shape[:2]) > config.max_proxy_mask_bytes
    ):
        raise MaterialsError("SAM2 proposal count or decoded/lifted mask bytes exceed budget")
    proposals = []
    for record in records:
        check_cancelled(cancelled)
        mask = _decode_rle(record["segmentation"], image.shape[:2])
        iou, stability = float(record["predicted_iou"]), float(record["stability_score"])
        if not math.isfinite(iou) or not math.isfinite(stability):
            raise MaterialsError("SAM2 geometric scores must be finite")
        proposals.append(Proposal(mask, float(np.clip(iou, 0, 1)), float(np.clip(stability, 0, 1))))
    proposals.sort(
        key=lambda item: (-int(item.mask.sum()), -item.geometric_quality, hashlib.sha256(item.mask.tobytes()).hexdigest())
    )
    return tuple(proposals)
