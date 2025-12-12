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
    
    TODO - PHASE 2 IMPLEMENTATION (Task 1: 24-32h):
    1. Research EfficientSAM model variants (S/Ti/distilled)
    2. Implement model loading and initialization
    3. Design prompt engineering for architectural scenes:
       - Grid-based prompts for comprehensive coverage
       - Edge-aware prompts for boundary refinement
       - Material-specific prompt templates
    4. Implement mask generation with quality filtering
    5. Integrate with Materials V2 property schema
    6. Add confidence scoring per mask
    7. Benchmark vs SegFormer-B5 on pool/kitchen scenes
    
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
        
        TODO: Implement model loading:
        - Load EfficientSAM checkpoint from cfg.efficientSAM_model
        - Initialize prompt encoder and mask decoder
        - Set up device placement and mixed precision
        - Validate model variant (S/Ti/distilled)
        """
        torch_ops.require_torch()
        self.cfg = cfg
        self.device = device
        
        # TODO: Replace with actual model loading
        raise NotImplementedError(
            "EfficientSAM backend is a Phase 2 stub. "
            "Implementation required: model loading, prompt engineering, mask generation. "
            "See PHASE2_IMPLEMENTATION_GUIDE.md for details."
        )
    
    def predict(self, rgb: "torch_ops.torch.Tensor") -> Dict[str, "torch_ops.torch.Tensor"]:
        """Generate material masks using EfficientSAM prompting.
        
        TODO: Implement mask generation:
        1. Preprocess RGB to EfficientSAM input format
        2. Generate grid-based prompts for scene coverage
        3. Run EfficientSAM inference with prompts
        4. Post-process masks (resize, soften, threshold)
        5. Map SAM masks to material buckets using CLIP classifier
        6. Return Dict[material_name, mask_tensor]
        
        Expected workflow:
            prompts = self._generate_architectural_prompts(rgb)
            sam_masks = self._run_efficientSAM(rgb, prompts)
            material_masks = self._classify_masks_with_CLIP(rgb, sam_masks)
            return material_masks
        """
        raise NotImplementedError("EfficientSAM predict() - Phase 2 stub")
    
    def _generate_architectural_prompts(self, rgb: "torch_ops.torch.Tensor") -> List[Dict]:
        """Generate prompts optimized for architectural scenes.
        
        TODO: Implement prompt engineering:
        - Grid-based point prompts for uniform coverage
        - Edge-aware box prompts for structural elements
        - Material-specific prompt templates (water=low points, sky=top region)
        - Adaptive prompt density based on scene complexity
        """
        raise NotImplementedError("Prompt engineering - Phase 2")
    
    def _classify_masks_with_CLIP(
        self,
        rgb: "torch_ops.torch.Tensor",
        sam_masks: List["torch_ops.torch.Tensor"]
    ) -> Dict[str, "torch_ops.torch.Tensor"]:
        """Classify SAM masks using CLIP zero-shot classification.
        
        TODO: Integrate with CLIPMaterialClassifier (materials_v2.py)
        - Extract region features for each SAM mask
        - Run CLIP zero-shot classification
        - Group masks by material type
        - Merge overlapping masks for same material
        """
        raise NotImplementedError("CLIP classification integration - Phase 2")


def create_material_segmenter(seg_cfg, device: "torch_ops.torch.device") -> Optional[MaterialSegmenter]:
    """Factory for material segmenter."""
    backend = (seg_cfg.backend or "auto").lower()

    if backend in ("none", "off", "disabled"):
        return None

    # auto selection: prefer ONNX if provided, else segformer if allowed, else heuristic
    if backend == "auto":
        if seg_cfg.onnx_model_path:
            backend = "onnx"
        elif seg_cfg.segformer_model or seg_cfg.allow_downloads:
            backend = "segformer"
        else:
            backend = "heuristic"

    if backend == "onnx":
        return OnnxMaterialSegmenter(seg_cfg, device)
    if backend == "segformer":
        return SegFormerAdekMaterialSegmenter(seg_cfg, device)
    if backend == "heuristic":
        return HeuristicMaterialSegmenter(seg_cfg, device)
    if backend == "efficientSAM":
        return EfficientSAMSegmenter(seg_cfg, device)

    if backend == "sam_clip":
        raise RuntimeError("sam_clip backend is a placeholder in V2 scaffold. Use onnx/segformer/heuristic for now.")

    raise ValueError(f"Unknown segmentation backend: {backend}")
