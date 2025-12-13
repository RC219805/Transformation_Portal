from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import torch_ops


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_model(path: Path, expected_sha256: Optional[str]) -> None:
    if not expected_sha256:
        return
    got = sha256_file(path)
    if got.lower() != expected_sha256.lower():
        raise RuntimeError(f"Model SHA256 mismatch. expected={expected_sha256} got={got}")


class Upscaler:
    def __init__(self, cfg, device: "torch_ops.torch.device"):
        torch_ops.require_torch()
        self.cfg = cfg
        self.device = device
        self.scale = int(getattr(cfg, "upscale", 4))

    def upscale(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
        raise NotImplementedError


class NoneUpscaler(Upscaler):
    def upscale(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
        # Use bicubic; if you need Lanczos quality, consider CPU cv2 resize explicitly.
        _, _, h, w = rgb.shape
        return torch_ops.resize(rgb, (h * self.scale, w * self.scale), mode="bicubic", autocast=True).clamp(0.0, 1.0)


class TorchUpscaler(Upscaler):
    """Torch-based upscaling using torchvision (safe alternative to Real-ESRGAN)."""
    def __init__(self, cfg, device: "torch_ops.torch.device"):
        super().__init__(cfg, device)
        try:
            import torch
            from torchvision.transforms import functional as TF  # type: ignore
        except Exception as e:
            raise RuntimeError("torchvision is required for torch upscaling") from e
        self.TF = TF
        self.tile_size = getattr(cfg, "upscale_tile_size", 0)
        self.tile_overlap = getattr(cfg, "upscale_tile_overlap", 64)
        
    def upscale(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
        """High-quality bicubic upscaling with optional tiling for memory efficiency."""
        _, _, h, w = rgb.shape
        
        # Use tiled upscaling for large images or when explicitly requested
        if self.tile_size > 0 and (h > self.tile_size or w > self.tile_size):
            return self._upscale_tiled(rgb)
        else:
            return self._upscale_full(rgb)
    
    def _upscale_full(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
        """Full-image upscaling (original behavior)."""
        _, _, h, w = rgb.shape
        target_h, target_w = h * self.scale, w * self.scale
        
        upscaled = self.TF.resize(
            rgb, 
            [target_h, target_w], 
            interpolation=self.TF.InterpolationMode.BICUBIC,
            antialias=True
        )
        return upscaled.clamp(0.0, 1.0)
    
    def _upscale_tiled(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
        """Memory-efficient tiled upscaling for large images."""
        import torch
        
        b, c, h, w = rgb.shape
        tile_size = self.tile_size
        overlap = self.tile_overlap
        scale = self.scale
        
        # Output dimensions
        out_h, out_w = h * scale, w * scale
        out = torch.zeros((b, c, out_h, out_w), dtype=torch.float32, device=self.device)
        weight = torch.zeros((b, c, out_h, out_w), dtype=torch.float32, device=self.device)
        
        # Compute tile grid
        y_starts = list(range(0, h, tile_size - overlap))
        x_starts = list(range(0, w, tile_size - overlap))
        
        # Process tiles with weighted blending
        for i, y0 in enumerate(y_starts):
            for j, x0 in enumerate(x_starts):
                # Input tile bounds with overlap
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                
                # Extract tile
                tile_in = rgb[:, :, y0:y1, x0:x1]
                
                # Upscale tile
                tile_h, tile_w = y1 - y0, x1 - x0
                tile_out = self.TF.resize(
                    tile_in,
                    [tile_h * scale, tile_w * scale],
                    interpolation=self.TF.InterpolationMode.BICUBIC,
                    antialias=True
                ).clamp(0.0, 1.0)
                
                # Output tile bounds
                out_y0, out_x0 = y0 * scale, x0 * scale
                out_y1, out_x1 = y1 * scale, x1 * scale
                
                # Determine which edges to fade based on tile position
                is_first_row = (i == 0)
                is_last_row = (i == len(y_starts) - 1)
                is_first_col = (j == 0)
                is_last_col = (j == len(x_starts) - 1)
                
                # Create blend weight for this tile
                tile_weight = self._create_positional_blend_mask(
                    tile_out.shape, 
                    overlap * scale,
                    fade_top=not is_first_row,
                    fade_bottom=not is_last_row,
                    fade_left=not is_first_col,
                    fade_right=not is_last_col
                )
                
                # Accumulate with weights
                out[:, :, out_y0:out_y1, out_x0:out_x1] += tile_out * tile_weight
                weight[:, :, out_y0:out_y1, out_x0:out_x1] += tile_weight
        
        # Normalize by accumulated weights
        out = out / (weight + 1e-8)
        
        return out
    
    def _create_positional_blend_mask(
        self, 
        shape: Tuple[int, ...], 
        overlap: int,
        fade_top: bool = True,
        fade_bottom: bool = True,
        fade_left: bool = True,
        fade_right: bool = True
    ) -> "torch_ops.torch.Tensor":
        """Create blend mask with selective edge fading based on tile position."""
        import torch
        
        b, c, h, w = shape
        mask = torch.ones((b, c, h, w), dtype=torch.float32, device=self.device)
        
        # Only apply feathering if we have actual overlap
        if overlap > 0 and overlap < min(h, w):
            fade = torch.linspace(0, 1, overlap, device=self.device)
            
            # Apply fade only on edges that overlap with other tiles
            if fade_top and h > overlap:
                mask[:, :, :overlap, :] *= fade.view(-1, 1)
            if fade_bottom and h > overlap:
                mask[:, :, -overlap:, :] *= fade.flip(0).view(-1, 1)
            if fade_left and w > overlap:
                mask[:, :, :, :overlap] *= fade.view(1, -1)
            if fade_right and w > overlap:
                mask[:, :, :, -overlap:] *= fade.flip(0).view(1, -1)
        
        return mask
    
    def _create_blend_mask(self, shape: Tuple[int, ...], overlap: int) -> "torch_ops.torch.Tensor":
        """Create blend mask for seamless tile merging with linear feathering."""
        import torch
        
        b, c, h, w = shape
        mask = torch.ones((b, c, h, w), dtype=torch.float32, device=self.device)
        
        # Only apply feathering if we have actual overlap
        if overlap > 0 and overlap < min(h, w):
            fade = torch.linspace(0, 1, overlap, device=self.device)
            
            # Apply fade on all edges for consistent blending
            # Top edge
            mask[:, :, :overlap, :] *= fade.view(-1, 1)
            # Bottom edge  
            mask[:, :, -overlap:, :] *= fade.flip(0).view(-1, 1)
            # Left edge
            mask[:, :, :, :overlap] *= fade.view(1, -1)
            # Right edge
            mask[:, :, :, -overlap:] *= fade.flip(0).view(1, -1)
        
        return mask


class OnnxUpscaler(Upscaler):
    def __init__(self, cfg, device: "torch_ops.torch.device"):
        super().__init__(cfg, device)
        if not cfg.model_path:
            raise ValueError("ONNX backend requires cfg.model_path to local .onnx")
        verify_model(Path(cfg.model_path), cfg.model_sha256)
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError("onnxruntime is required for ONNX upscaling") from e

        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(cfg.model_path), sess_options=sess_opt, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def upscale(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
        # ONNX expects NCHW float32 0..1
        inp = rgb.detach().to("cpu", dtype=torch_ops.torch.float32).numpy()
        out = self.sess.run([self.out_name], {self.in_name: inp})[0]  # NCHW
        out_t = torch_ops.torch.from_numpy(out).to(device=self.device, dtype=torch_ops.torch.float32)
        return out_t.clamp(0.0, 1.0)


def create_upscaler(cfg, device: "torch_ops.torch.device") -> Upscaler:
    backend = str(getattr(cfg, "upscaler_backend", "torch")).lower()
    if backend == "none":
        return NoneUpscaler(cfg, device)
    if backend == "onnx":
        return OnnxUpscaler(cfg, device)
    if backend in ("torch", "torchvision"):
        return TorchUpscaler(cfg, device)
    # Legacy support: map realesrgan to torch backend
    if backend == "realesrgan":
        import warnings
        warnings.warn(
            "RealESRGAN backend is deprecated due to CVE-2024-27763. "
            "Using torch backend instead. See lux_depth_v2/SECURITY.md",
            DeprecationWarning,
            stacklevel=2
        )
        return TorchUpscaler(cfg, device)
    raise ValueError(f"Unknown upscaler backend: {backend}")
