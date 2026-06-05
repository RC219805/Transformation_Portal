#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

try:
    import depth_pro
except ImportError:  # pragma: no cover - optional model runtime
    depth_pro = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DepthProVerifier")


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def verify_depth_pro() -> bool:
    logger.info("--- Starting Depth Pro Verification ---")

    if depth_pro is None:
        logger.error("❌ depth_pro package is not installed. Install the Depth Pro runtime first.")
        return False

    device = pick_device()
    logger.info(f"ℹ️  Using device: {device}")

    ckpt = Path("./checkpoints/depth_pro.pt").resolve()
    if not ckpt.exists():
        logger.error(f"❌ Checkpoint not found: {ckpt}")
        logger.error(
            "   Download: curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt"
        )
        return False
    logger.info(f"ℹ️  Checkpoint: {ckpt} ({ckpt.stat().st_size/1e9:.2f} GB)")

    logger.info("⏳ Loading model + weights...")
    try:
        # Uses Depth Pro's supported API (no DepthProConfig required)
        model, transform = depth_pro.create_model_and_transforms()
        model = model.to(device).eval()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model ready: {model.__class__.__name__} | {n_params/1e6:.1f}M params")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

    logger.info("⏳ Running dummy inference...")
    try:
        dummy = torch.rand(1, 3, 512, 512, device=device, dtype=torch.float32)

        # Some Depth Pro transforms expect PIL images; if this errors, we skip it for dummy noise.
        try:
            dummy_in = transform(dummy) if callable(transform) else dummy
        except Exception:
            dummy_in = dummy

        with torch.no_grad():
            out = model.infer(dummy_in)

        depth = out["depth"]
        logger.info(f"✅ Inference successful. Output shape: {tuple(depth.shape)}")
        return True
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if verify_depth_pro() else 1)
