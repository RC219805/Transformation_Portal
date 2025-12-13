from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import torch_ops


class MaterialSegmenter:
    """Interface for material mask prediction."""

    def predict(self, rgb: "torch_ops.torch.Tensor") -> Dict[str, "torch_ops.torch.Tensor"]:
        """Return dict: surface -> 1x1xHxW mask (0..1)."""
        raise NotImplementedError


def _resize_long_side(rgb: "torch_ops.torch.Tensor", long_side: int) -> Tuple["torch_ops.torch.Tensor", float]:
    b, c, h, w = rgb.shape
    ls = max(h, w)
    if ls <= long_side:
        return rgb, 1.0
    scale = float(long_side) / float(ls)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return torch_ops.resize(rgb, (nh, nw), mode="bilinear", autocast=True), scale


def _soften_mask(mask: "torch_ops.torch.Tensor", sigma_px: float) -> "torch_ops.torch.Tensor":
    if sigma_px <= 0:
        return mask
    return torch_ops.gaussian_blur(mask, sigma_px, autocast=True).clamp(0.0, 1.0)


class HeuristicMaterialSegmenter(MaterialSegmenter):
    """Fast, dependency-free heuristic masks.

    This is a fallback. Quality improves substantially with an ML backend.
    """

    def __init__(self, cfg, device: "torch_ops.torch.device"):
        torch_ops.require_torch()
        self.cfg = cfg
        self.device = device

    def predict(self, rgb: "torch_ops.torch.Tensor") -> Dict[str, "torch_ops.torch.Tensor"]:
        torch_ops.require_torch()
        # rgb: 1x3xHxW, 0..1
        r = rgb[:, 0:1].clamp(0, 1)
        g = rgb[:, 1:2].clamp(0, 1)
        b = rgb[:, 2:3].clamp(0, 1)
        l = torch_ops.luma(rgb)

        # crude saturation proxy
        mx = torch_ops.torch.maximum(torch_ops.torch.maximum(r, g), b)
        mn = torch_ops.torch.minimum(torch_ops.torch.minimum(r, g), b)
        sat = (mx - mn) / (mx + 1e-4)

        masks: Dict[str, torch_ops.torch.Tensor] = {}

        # sky: blue-dominant, reasonably bright, moderate saturation
        sky = ((b > r + 0.08) & (b > g + 0.05) & (l > 0.25)).to(torch_ops.torch.float32) * (sat.clamp(0, 1) ** 0.5)
        masks["sky"] = sky.clamp(0, 1)

        # foliage: green-dominant with some saturation
        foliage = ((g > r + 0.05) & (g > b + 0.02) & (sat > 0.15)).to(torch_ops.torch.float32) * (sat.clamp(0, 1))
        masks["foliage"] = foliage.clamp(0, 1)

        # wood: warm-ish hues, medium saturation, mid brightness
        wood = ((r > b + 0.03) & (g > b + 0.01) & (sat > 0.10) & (l > 0.15) & (l < 0.85)).to(torch_ops.torch.float32)
        wood = wood * (0.6 + 0.4 * sat.clamp(0, 1))
        masks["wood"] = wood.clamp(0, 1)

        # metal: low saturation, mid-high brightness
        metal = ((sat < 0.18) & (l > 0.25)).to(torch_ops.torch.float32) * (l.clamp(0, 1) ** 0.7)
        masks["metal"] = metal.clamp(0, 1)

        # glass: very low saturation + bright highlights (specular), but not sky
        glass = ((sat < 0.14) & (l > 0.55)).to(torch_ops.torch.float32) * (l.clamp(0, 1) ** 1.2) * (1.0 - masks["sky"])
        masks["glass"] = glass.clamp(0, 1)

        # stone: low saturation, mid brightness, not metal/glass
        stone = ((sat < 0.22) & (l > 0.15) & (l < 0.80)).to(torch_ops.torch.float32)
        stone = stone * (1.0 - torch_ops.torch.maximum(masks["metal"], masks["glass"]))
        masks["stone"] = stone.clamp(0, 1)

        # soften
        sigma = float(getattr(self.cfg, "soften_sigma_px", 2.0))
        for k in list(masks.keys()):
            masks[k] = _soften_mask(masks[k], sigma)

        # confidence threshold
        min_c = float(getattr(self.cfg, "min_confidence", 0.25))
        for k in list(masks.keys()):
            masks[k] = torch_ops.torch.where(masks[k] >= min_c, masks[k], torch_ops.torch.zeros_like(masks[k]))

        return masks


class OnnxMaterialSegmenter(MaterialSegmenter):
    """ONNX Runtime segmentation backend.

    Expected model:
      - input: 1x3xHxW float32 RGB in 0..1
      - output: either:
          a) 1xCxHxW logits/probabilities for classes
          b) 1xHxW class ids (int64)
    Labels:
      - provide a JSON mapping of class index to surface name, or fallback to cfg.surfaces order.
    """

    def __init__(self, cfg, device: "torch_ops.torch.device"):
        torch_ops.require_torch()
        self.cfg = cfg
        self.device = device
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError("onnxruntime is required for ONNX material segmentation") from e

        if not cfg.onnx_model_path:
            raise ValueError("segmentation.onnx_model_path is required for backend=onnx")

        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(cfg.onnx_model_path), sess_options=so, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        self.id_to_surface: Dict[int, str] = {}
        if cfg.onnx_labels_path and Path(cfg.onnx_labels_path).exists():
            raw = json.loads(Path(cfg.onnx_labels_path).read_text(encoding="utf-8"))
            # allow {"0":"wood"} or {"wood":0}
            if all(isinstance(k, str) and k.isdigit() for k in raw.keys()):
                self.id_to_surface = {int(k): str(v) for k, v in raw.items()}
            else:
                # invert
                self.id_to_surface = {int(v): str(k) for k, v in raw.items()}

    def predict(self, rgb: "torch_ops.torch.Tensor") -> Dict[str, "torch_ops.torch.Tensor"]:
        torch_ops.require_torch()
        device = rgb.device
        rgb_in, scale = _resize_long_side(rgb, int(self.cfg.input_long_side))

        inp = rgb_in.detach().to("cpu", dtype=torch_ops.torch.float32).numpy()
        out = self.sess.run([self.out_name], {self.in_name: inp})[0]

        # Convert output to per-surface probability masks
        masks: Dict[str, torch_ops.torch.Tensor] = {}
        if out.ndim == 3:
            # 1xHxW class ids
            cls = torch_ops.torch.from_numpy(out).to(device=device)
            if cls.dtype not in (torch_ops.torch.int64, torch_ops.torch.int32):
                cls = cls.long()
            cls = cls.unsqueeze(1)  # 1x1xHxW
            for idx, surf in self.id_to_surface.items():
                masks[surf] = (cls == int(idx)).to(torch_ops.torch.float32)
        elif out.ndim == 4:
            # 1xCxHxW logits or probs
            logits = torch_ops.torch.from_numpy(out).to(device=device, dtype=torch_ops.torch.float32)
            probs = torch_ops.torch.softmax(logits, dim=1)
            # map ids
            if self.id_to_surface:
                for idx, surf in self.id_to_surface.items():
                    if idx < probs.shape[1]:
                        masks[surf] = probs[:, idx:idx+1]
            else:
                # fallback: assume cfg.surfaces order matches channels
                for i, surf in enumerate(getattr(self.cfg, "surfaces", ())):
                    if i < probs.shape[1]:
                        masks[surf] = probs[:, i:i+1]
        else:
            raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")

        # resize masks back to original if needed
        if scale != 1.0:
            _, _, h0, w0 = rgb.shape
            for k in list(masks.keys()):
                masks[k] = torch_ops.resize(masks[k], (h0, w0), mode="bilinear", autocast=True)

        # soften and threshold
        sigma = float(getattr(self.cfg, "soften_sigma_px", 2.0))
        min_c = float(getattr(self.cfg, "min_confidence", 0.25))
        for k in list(masks.keys()):
            masks[k] = _soften_mask(masks[k].clamp(0.0, 1.0), sigma)
            masks[k] = torch_ops.torch.where(masks[k] >= min_c, masks[k], torch_ops.torch.zeros_like(masks[k]))

        return masks


class SegFormerAdekMaterialSegmenter(MaterialSegmenter):
    """Practical 'advanced' backend: scene parsing -> material proxy masks.

    Uses a SegFormer ADE20K scene parsing model and maps semantic labels to our material buckets.
    This is *not* true material segmentation, but is often strong enough for interior/exterior real-estate images.

    Requires: transformers, torch, (optionally) huggingface_hub if allow_downloads=True.
    """

    def __init__(self, cfg, device: "torch_ops.torch.device"):
        torch_ops.require_torch()
        self.cfg = cfg
        self.device = device

        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor  # type: ignore
        except Exception as e:
            raise RuntimeError("transformers is required for segformer backend") from e

        # PRODUCTION DEFAULT: SegFormer-B5 for highest quality material segmentation
        model_id = cfg.segformer_model or "nvidia/segformer-b5-finetuned-ade-640-640"
        
        # block downloads unless explicitly allowed
        if not cfg.allow_downloads and not (Path(model_id).exists() and Path(model_id).is_dir()):
            raise RuntimeError(
                "segmentation.allow_downloads=False and segformer_model is not a local directory. "
                "Provide a local model dir or set allow_downloads=True."
            )

        # Use local_files_only if downloads not allowed
        download_kwargs = {"local_files_only": not cfg.allow_downloads}
        
        # Only add revision for remote models when specified (avoid cache issues)
        if cfg.segformer_revision and cfg.allow_downloads:
            download_kwargs["revision"] = cfg.segformer_revision
        
        self.processor = SegformerImageProcessor.from_pretrained(model_id, **download_kwargs)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id, **download_kwargs)
        self.model.to(device)
        self.model.eval()

        # label mapping from model
        id2label = self.model.config.id2label  # type: ignore
        self.labels = {int(k): str(v).lower() for k, v in id2label.items()}

        # Build token sets for mapping semantics -> our buckets
        self.bucket_rules = {
            "glass":   ["window", "windowpane", "glass", "mirror", "screen"],
            "wood":    ["wood", "door", "cabinet", "table", "chair", "desk", "floor", "stairs", "shelf"],
            "metal":   ["sink", "faucet", "rail", "refrigerator", "oven", "microwave", "stove", "dishwasher"],
            "stone":   ["wall", "concrete", "brick", "tile", "counter", "countertop", "ceiling", "pavement"],
            "sky":     ["sky"],
            "foliage": ["tree", "plant", "grass", "foliage"],
        }

        # Precompute class index lists for each bucket
        self.bucket_ids: Dict[str, List[int]] = {k: [] for k in self.bucket_rules}
        for idx, name in self.labels.items():
            for bucket, toks in self.bucket_rules.items():
                if any(tok in name for tok in toks):
                    self.bucket_ids[bucket].append(idx)

    def predict(self, rgb: "torch_ops.torch.Tensor") -> Dict[str, "torch_ops.torch.Tensor"]:
        torch_ops.require_torch()
        device = rgb.device
        
        # Use context manager instead of decorator to avoid import-time issues
        with torch_ops.torch.no_grad():
            rgb_in, scale = _resize_long_side(rgb, int(self.cfg.input_long_side))

            # processor expects PIL or numpy; use cpu numpy for preprocessing
            np_img = (rgb_in[0].permute(1,2,0).clamp(0,1).to("cpu").numpy() * 255.0).astype(np.uint8)

            inputs = self.processor(images=np_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits  # 1xCxH'xW'
            probs = torch_ops.torch.softmax(logits.to(dtype=torch_ops.torch.float32), dim=1)

            # upsample to rgb_in resolution
            probs = torch_ops.resize(probs, (rgb_in.shape[2], rgb_in.shape[3]), mode="bilinear", autocast=True)

            masks: Dict[str, torch_ops.torch.Tensor] = {}
            for bucket, ids in self.bucket_ids.items():
                if not ids:
                    continue
                # sum probabilities of relevant classes
                idx = torch_ops.torch.tensor(ids, device=device, dtype=torch_ops.torch.long)
                m = probs.index_select(1, idx).sum(dim=1, keepdim=True)
                masks[bucket] = m.clamp(0.0, 1.0)

            # resize masks back to original if needed
            if scale != 1.0:
                _, _, h0, w0 = rgb.shape
                for k in list(masks.keys()):
                    masks[k] = torch_ops.resize(masks[k], (h0, w0), mode="bilinear", autocast=True)

            sigma = float(getattr(self.cfg, "soften_sigma_px", 2.0))
            min_c = float(getattr(self.cfg, "min_confidence", 0.25))
            for k in list(masks.keys()):
                masks[k] = _soften_mask(masks[k].clamp(0.0, 1.0), sigma)
                masks[k] = torch_ops.torch.where(masks[k] >= min_c, masks[k], torch_ops.torch.zeros_like(masks[k]))

            return masks


class EfficientSAMSegmenter(MaterialSegmenter):
    """EfficientSAM-based material segmentation (PHASE 2 - STUB).
    
    Provides high-precision boundary detection using EfficientSAM prompting.
    Expected 60-80% improvement in boundary precision over SegFormer-B5.
    
    This is an architectural stub for Phase 2 implementation.
    
    Phase 2 Implementation Tasks (24-32h):
    - EfficientSAM model loading and initialization
    - Prompt engineering for architectural scenes (grid-based, edge-aware)
    - Mask generation with quality filtering and confidence scoring
    - Integration with Materials V2 property schema
    - Benchmarking vs SegFormer-B5
    
    Expected API:
        >>> segmenter = EfficientSAMSegmenter(cfg, device)
        >>> masks = segmenter.predict(rgb_tensor)  # Dict[str, Tensor]
        >>> # masks["water"], masks["stone"], etc.
    
    Integration Points:
        - config.SegmentationConfig.backend = "efficientSAM"
        - config.SegmentationConfig.efficientSAM_model = "path/to/model"
        - materials_v2.MaterialsV2Engine (via create_material_segmenter)
    """
    
    def __init__(self, cfg, device: "torch_ops.torch.device"):
        """Initialize EfficientSAM segmenter.
        
        Phase 2 stub - requires implementation of:
        model loading, prompt encoder, mask decoder, mixed precision setup
        """
        torch_ops.require_torch()
        self.cfg = cfg
        self.device = device
        
        raise NotImplementedError(
            "EfficientSAM backend is a Phase 2 stub. "
            "See PHASE2_IMPLEMENTATION_GUIDE.md for implementation details."
        )
    
    def predict(self, rgb: "torch_ops.torch.Tensor") -> Dict[str, "torch_ops.torch.Tensor"]:
        """Generate material masks using EfficientSAM prompting.
        
        Phase 2 workflow:
        1. Generate architectural prompts (grid-based, edge-aware)
        2. Run EfficientSAM inference
        3. Post-process and classify masks with CLIP
        4. Return material-specific masks
        """
        raise NotImplementedError("EfficientSAM predict() - Phase 2 stub")
    
    def _generate_architectural_prompts(self, rgb: "torch_ops.torch.Tensor") -> List[Dict]:
        """Generate prompts optimized for architectural scenes.
        
        Phase 2: Grid-based, edge-aware, and material-specific prompts.
        """
        raise NotImplementedError("Prompt engineering - Phase 2")
    
    def _classify_masks_with_CLIP(
        self,
        rgb: "torch_ops.torch.Tensor",
        sam_masks: List["torch_ops.torch.Tensor"]
    ) -> Dict[str, "torch_ops.torch.Tensor"]:
        """Classify SAM masks using CLIP zero-shot classification.
        
        Phase 2: Integrate with CLIPMaterialClassifier for mask classification.
        - Merge overlapping masks for same material
        """
        raise NotImplementedError("CLIP classification integration - Phase 2")


# -------------------------------------------------------------------------
# EfficientSAM V3 Fusion Integration
# -------------------------------------------------------------------------

# Conservative edge refinement class list
EDGE_REFINEMENT_CLASSES = {"glass", "water", "foliage"}


class FusedMaterialSegmenter(MaterialSegmenter):
    """
    Material Segmentation V3: SegFormer + EfficientSAM fusion.

    Wraps a base segmenter (typically SegFormer) and optionally refines
    edges for specific material classes using EfficientSAM.

    Stage 3 behavior:
    - When fusion_mode != NONE and a refinement provider is available,
      selected classes are refined via EfficientSAM and fused using
      IoU gating + confidence-weighted blending.
    - Falls back gracefully to base masks on any failure.
    - Emits fusion statistics for quality monitoring.
    """

    def __init__(
        self,
        base_segmenter: MaterialSegmenter,
        cfg,
        device: "torch_ops.torch.device",
        refinement_provider=None,
    ):
        self.base_segmenter = base_segmenter
        self.cfg = cfg
        self.device = device
        self.refinement_provider = refinement_provider
        self.fusion_stats: Dict[str, Dict[str, float]] = {}

    def predict(self, rgb: "torch_ops.torch.Tensor") -> Dict[str, "torch_ops.torch.Tensor"]:
        """
        Generate material masks with optional edge refinement.

        Returns base masks if:
        - fusion_mode is NONE
        - refinement_provider is None
        - no classes are in EDGE_REFINEMENT_CLASSES
        """
        torch_ops.require_torch()
        from .segmentation_fusion import FusionConfig, FusionMode, fuse_masks

        # Always get base masks first
        base_masks = self.base_segmenter.predict(rgb)

        # Check if fusion is enabled
        fusion_mode = getattr(self.cfg, "fusion_mode", FusionMode.NONE)
        if fusion_mode == FusionMode.NONE or self.refinement_provider is None:
            return base_masks

        # Build fusion config from segmentation config
        fusion_cfg = FusionConfig(
            mode=fusion_mode,
            min_iou=float(getattr(self.cfg, "fusion_min_iou", 0.30)),
            core_thresh=float(getattr(self.cfg, "fusion_core_thresh", 0.70)),
            edge_low=float(getattr(self.cfg, "fusion_edge_low", 0.20)),
            edge_high=float(getattr(self.cfg, "fusion_edge_high", 0.70)),
            alpha_edge=float(getattr(self.cfg, "fusion_alpha_edge", 0.70)),
            alpha_core=float(getattr(self.cfg, "fusion_alpha_core", 0.30)),
            clamp=True,
        )

        # Refine selected classes
        fused_masks = {}
        self.fusion_stats = {}

        for material_class, base_mask in base_masks.items():
            # Only refine targeted classes
            if material_class not in EDGE_REFINEMENT_CLASSES:
                fused_masks[material_class] = base_mask
                continue

            try:
                # Get refined mask from provider
                refined_mask = self.refinement_provider.get_refined_mask(
                    rgb, base_mask, material_class
                )

                if refined_mask is None:
                    # Provider unavailable or failed
                    fused_masks[material_class] = base_mask
                    self.fusion_stats[material_class] = {
                        "iou_base_vs_refined": 0.0,
                        "fusion_applied": 0.0,
                    }
                    continue

                # Convert masks to numpy for fusion
                base_np = base_mask[0, 0].detach().to("cpu").numpy().astype(np.float32)
                refined_np = refined_mask[0, 0].detach().to("cpu").numpy().astype(np.float32)

                # Apply fusion with IoU gating
                fused_np, stats = fuse_masks(base_np, refined_np, fusion_cfg)

                # Convert back to torch
                fused_tensor = (
                    torch_ops.torch.from_numpy(fused_np)
                    .to(device=self.device, dtype=torch_ops.torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                fused_masks[material_class] = fused_tensor
                self.fusion_stats[material_class] = stats

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Fusion failed for %s: %s. Falling back to base mask.",
                    material_class,
                    e,
                )
                fused_masks[material_class] = base_mask
                self.fusion_stats[material_class] = {
                    "iou_base_vs_refined": 0.0,
                    "fusion_applied": 0.0,
                }

        return fused_masks


def create_material_segmenter(seg_cfg, device: "torch_ops.torch.device") -> Optional[MaterialSegmenter]:
    """Factory for material segmenter with V3 fusion support."""
    from .config import SegmentationBackend, FusionMode

    # Handle legacy backend string for backward compatibility
    backend = (seg_cfg.backend or "auto").lower()

    if backend in ("none", "off", "disabled"):
        return None

    # Check V3 backend setting (typed enum)
    backend_v3 = getattr(seg_cfg, "backend_v3", None)
    use_fusion = getattr(seg_cfg, "use_efficientsam_for_edges", False)
    fusion_mode = getattr(seg_cfg, "fusion_mode", FusionMode.NONE)

    # Determine base segmenter
    base_segmenter = None

    # V3 path: use backend_v3 if set and not SEGFORMER-only
    if backend_v3 is not None and backend_v3 != SegmentationBackend.SEGFORMER:
        if backend_v3 == SegmentationBackend.EFFICIENTSAM:
            base_segmenter = EfficientSAMSegmenter(seg_cfg, device)
        elif backend_v3 == SegmentationBackend.FUSED:
            # FUSED means: SegFormer base + EfficientSAM refinement
            base_segmenter = SegFormerAdekMaterialSegmenter(seg_cfg, device)
        else:
            raise ValueError(f"Unknown backend_v3: {backend_v3}")
    else:
        # Legacy path: use backend string
        if backend == "auto":
            if seg_cfg.onnx_model_path:
                backend = "onnx"
            elif seg_cfg.segformer_model or seg_cfg.allow_downloads:
                backend = "segformer"
            else:
                backend = "heuristic"

        if backend == "onnx":
            base_segmenter = OnnxMaterialSegmenter(seg_cfg, device)
        elif backend == "segformer":
            base_segmenter = SegFormerAdekMaterialSegmenter(seg_cfg, device)
        elif backend == "heuristic":
            base_segmenter = HeuristicMaterialSegmenter(seg_cfg, device)
        elif backend in ("efficientsam", "efficientsam_backend", "efficientsam_v3"):
            base_segmenter = EfficientSAMSegmenter(seg_cfg, device)
        elif backend == "sam_clip":
            raise RuntimeError("sam_clip backend is a placeholder. Use onnx/segformer/heuristic.")
        else:
            raise ValueError(f"Unknown segmentation backend: {backend}")

    # V3 fusion wrapper (only if explicitly requested)
    if backend_v3 == SegmentationBackend.FUSED or use_fusion:
        # Try to create EfficientSAM refinement provider
        refinement_provider = None
        try:
            from .backends.efficientsam_backend import EfficientSAMBackend
            from .backends.refinement_provider import EfficientSAMRefinementProvider

            esam_backend = EfficientSAMBackend(
                model_name=getattr(seg_cfg, "efficientSAM_model", "efficientsam_ti_vit_s"),
                device="cpu",  # EfficientSAM runs on CPU for Stage 3
                lazy_load=True,
            )

            if esam_backend.available:
                refinement_provider = EfficientSAMRefinementProvider(esam_backend, device)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "EfficientSAM refinement provider unavailable: %s. Fusion disabled.", e
            )

        return FusedMaterialSegmenter(base_segmenter, seg_cfg, device, refinement_provider)

    return base_segmenter
