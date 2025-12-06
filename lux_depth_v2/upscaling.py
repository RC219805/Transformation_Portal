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


class RealESRGANUpscaler(Upscaler):
    def __init__(self, cfg, device: "torch_ops.torch.device"):
        super().__init__(cfg, device)
        if not cfg.model_path:
            raise ValueError("RealESRGAN backend requires cfg.model_path to local .pth")
        verify_model(Path(cfg.model_path), cfg.model_sha256)

        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            from realesrgan import RealESRGANer  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing Real-ESRGAN deps. Install: realesrgan basicsr") from e

        # Choose device for RealESRGANer
        dev = str(cfg.device).lower()
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=self.scale
        )

        self._er = RealESRGANer(
            scale=self.scale,
            model_path=str(cfg.model_path),
            model=model,
            tile=int(cfg.tile) if int(cfg.tile) > 0 else 0,
            tile_pad=int(cfg.tile_pad),
            pre_pad=0,
            half=bool(cfg.half),
            gpu_id=0 if dev == "cuda" else None,
        )

    def upscale(self, rgb: "torch_ops.torch.Tensor") -> "torch_ops.torch.Tensor":
        # Real-ESRGAN expects BGR uint8 HxWx3
        torch_ops.require_torch()
        inp = (rgb.detach().clamp(0,1)[0].permute(1,2,0).to("cpu").numpy() * 255.0 + 0.5).astype(np.uint8)
        inp = inp[..., ::-1]  # RGB->BGR
        out, _ = self._er.enhance(inp, outscale=self.scale)
        if out.ndim == 2:
            out = np.stack([out, out, out], axis=-1)
        out = out[..., ::-1]  # BGR->RGB
        out01 = (out.astype(np.float32) / 255.0).clip(0.0, 1.0)
        return torch_ops.to_torch_rgb(out01, device=self.device)


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
    backend = str(getattr(cfg, "upscaler_backend", "realesrgan")).lower()
    if backend == "none":
        return NoneUpscaler(cfg, device)
    if backend == "onnx":
        return OnnxUpscaler(cfg, device)
    if backend == "realesrgan":
        return RealESRGANUpscaler(cfg, device)
    raise ValueError(f"Unknown upscaler backend: {backend}")
