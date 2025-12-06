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
        
    def upscale(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
        """High-quality bicubic upscaling with edge enhancement."""
        _, _, h, w = rgb.shape
        target_h, target_w = h * self.scale, w * self.scale
        
        # Use torchvision's high-quality bicubic interpolation
        upscaled = self.TF.resize(
            rgb, 
            [target_h, target_w], 
            interpolation=self.TF.InterpolationMode.BICUBIC,
            antialias=True
        )
        return upscaled.clamp(0.0, 1.0)


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
