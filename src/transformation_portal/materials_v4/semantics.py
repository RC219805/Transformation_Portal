"""Local OpenCLIP experiment with masked microbatches and explicit rejection.

Scores are relative rankings, not calibrated material probabilities. This module
does not emit a calibration receipt or any automatic photographic-edit authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image

from .contracts import MaterialsError
from .proposal import Proposal, check_cancelled

CLIP_MODEL_NAME = "ViT-B-32"
SEMANTIC_SCORE_TYPE = "uncalibrated_clip_ranking_v1"
MATERIAL_PROMPTS = (
    ("sky", "a photograph of sky and clouds"),
    ("glass", "a photograph of a glass window surface"),
    ("water", "a photograph of water in a pool or ocean"),
    ("foliage", "a photograph of plant leaves and foliage"),
    ("wood", "a photograph of a wood surface"),
    ("stone", "a photograph of natural stone or concrete"),
    ("metal", "a photograph of a metal surface"),
    ("fabric", "a photograph of fabric upholstery or curtains"),
    ("stucco", "a photograph of stucco or plaster"),
    ("paint", "a photograph of a painted surface"),
    ("ceramic", "a photograph of ceramic or porcelain tile"),
    ("leather", "a photograph of leather upholstery"),
    ("unknown", "a photograph of an object made of another material"),
)
PREPROCESS_CONTRACT = {
    "model": CLIP_MODEL_NAME,
    "quick_gelu": True,
    "color": "encoded_srgb_uint8",
    "crop": "tight_mask_bbox",
    "outside_mask": "black",
    "resize": "bicubic_shortest_center_crop_224",
    "normalization": "openai_clip_mean_std",
    "score": "cosine_softmax_fixed_scale_20_uncalibrated",
}


@dataclass(frozen=True)
class SemanticObservation:
    label: str
    confidence: float
    cosine_similarity: float
    margin: float
    rejection: str | None


def classify_scores(similarities: np.ndarray, config: Any) -> SemanticObservation:
    values = np.asarray(similarities, dtype=np.float64)
    if values.shape != (len(MATERIAL_PROMPTS),) or not np.isfinite(values).all():
        raise MaterialsError("Semantic scores must be one finite value per prompt")
    logits = values * 20.0
    exp = np.exp(logits - logits.max())
    probabilities = exp / exp.sum()
    index = int(np.argmax(values))
    top = float(probabilities[index])
    ordered = np.sort(probabilities)
    margin = float(ordered[-1] - ordered[-2])
    label = MATERIAL_PROMPTS[index][0]
    reason = None
    if label == "unknown":
        reason = "unknown_prompt"
    elif float(values[index]) < config.min_similarity:
        reason = "below_absolute_similarity"
    elif top < config.min_top_probability:
        reason = "below_ranking_threshold"
    elif margin < config.min_margin:
        reason = "ambiguous_ranking"
    return SemanticObservation("unknown" if reason is not None else label, top, float(values[index]), margin, reason)


def masked_crop(image: np.ndarray, mask: np.ndarray) -> Image.Image | None:
    if mask.shape != image.shape[:2] or mask.dtype != bool:
        raise MaterialsError("Semantic mask must match proxy geometry")
    ys, xs = np.where(mask)
    if not len(ys):
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    crop = image[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = 0
    return Image.fromarray(crop)


def classify_proposals(
    image: np.ndarray,
    proposals: tuple[Proposal, ...],
    config: Any,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[SemanticObservation, ...]:
    check_cancelled(cancelled)
    if not proposals:
        return ()
    try:
        import open_clip
        import torch
        from safetensors.torch import load_file

        # Built-in architecture and bundled tokenizer only. No pretrained tags,
        # hub identifiers, tower downloads, or heuristic/error substitutions.
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=None,
            device=config.device,
            force_quick_gelu=True,
            pretrained_image=False,
            pretrained_text=False,
            image_interpolation="bicubic",
            image_resize_mode="shortest",
        )
        state = load_file(str(config.clip_checkpoint), device="cpu")
        model.load_state_dict(state, strict=True)
        del state
        model.eval()
        if str(next(model.parameters()).device).split(":", 1)[0] != config.device:
            raise MaterialsError("CLIP actual device differs from the frozen explicit device")
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        with torch.inference_mode():
            tokens = tokenizer([prompt for _, prompt in MATERIAL_PROMPTS]).to(config.device)
            text = model.encode_text(tokens)
            text = text / text.norm(dim=-1, keepdim=True)
            results: list[SemanticObservation] = []
            for start in range(0, len(proposals), config.classifier_batch_size):
                check_cancelled(cancelled)
                batch = proposals[start : start + config.classifier_batch_size]
                crops = [masked_crop(image, proposal.mask) for proposal in batch]
                active = [(index, crop) for index, crop in enumerate(crops) if crop is not None]
                observations = [SemanticObservation("unknown", 0.0, 0.0, 0.0, "empty_mask") for _ in batch]
                if active:
                    tensors = torch.stack([preprocess(crop) for _, crop in active]).to(config.device)
                    features = model.encode_image(tensors)
                    features = features / features.norm(dim=-1, keepdim=True)
                    similarities = (features @ text.T).detach().cpu().float().numpy()
                    for row, (index, _) in enumerate(active):
                        observations[index] = classify_scores(similarities[row], config)
                    del tensors, features, similarities
                results.extend(observations)
                check_cancelled(cancelled)
        return tuple(results)
    except MaterialsError:
        raise
    except Exception as exc:
        raise MaterialsError(f"Local CLIP classification failed without fallback: {exc}") from exc
