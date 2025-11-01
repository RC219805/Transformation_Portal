#!/usr/bin/env python3
"""
agx_batch_processor.py
Multithreaded AgX + Filmic tone mapping pipeline for architectural visualization.

New features:
- ThreadPoolExecutor batch processing (--workers)
- Per-image auto-exposure (--auto-exposure [logmean|median])
- Auto-exposure computed in scene-linear domain with a "key" (default 0.18)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Ensure module access
sys.path.append(str(Path(__file__).parent))

# Tonemapper module import
try:
    from tonemapper_agx_filmic import (
        apply_agx_ocio,
        apply_filmic_hable,
        list_ocio_views,
        guess_agx_view,
        srgb_to_linear,
        linear_to_srgb
    )
except ImportError as e:
    print(f"ERROR: tonemapper_agx_filmic not found: {e}")
    sys.exit(1)

# Optional OCIO
try:
    import PyOpenColorIO as ocio
    HAVE_OCIO = True
except ImportError:
    HAVE_OCIO = False

# Optional imageio for EXR
try:
    import imageio.v3 as iio
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False


# ------------------------------
# Logging Setup
# ------------------------------
log = logging.getLogger("AgXBatch")
log.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(ch)


# ------------------------------
# Utility Functions
# ------------------------------
def load_image(path: Path, to_linear: bool = False) -> np.ndarray:
    """Load an image to float32 array in range [0,1]."""
    suffix = path.suffix.lower()
    if suffix in (".exr", ".hdr"):
        if not HAVE_IMAGEIO:
            log.error("imageio required for EXR/HDR. Install with: pip install imageio")
            raise RuntimeError("imageio required for EXR/HDR")
        arr = iio.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        if to_linear:
            # EXR is usually already linear; keep as-is
            pass
        return arr

    img = Image.open(path)
    # Normalize orientation by loading then converting
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    if to_linear:
        arr = srgb_to_linear(arr)
    return arr


def save_image(arr: np.ndarray, path: Path, quality: int = 95, save_linear: bool = False):
    """Save array as JPG/PNG or EXR if requested."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if save_linear and path.suffix.lower() == ".exr":
        if not HAVE_IMAGEIO:
            log.error("imageio required for EXR output. Install with: pip install imageio")
            raise RuntimeError("imageio required for EXR output")
        iio.imwrite(path, arr.astype(np.float32))
        log.info(f"✓ Saved linear EXR: {path}")
        return

    arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8, "RGB")

    if path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(path, quality=quality, optimize=True)
    elif path.suffix.lower() == ".png":
        img.save(path, compress_level=6)
    else:
        img.save(path)
    log.info(f"✓ Saved: {path}")


# ------------------------------
# Auto-exposure helpers
# ------------------------------
def luminance(arr: np.ndarray) -> np.ndarray:
    """Return luminance per pixel (scene-linear array)."""
    # Rec.709 / Rec.2020-ish luma weights
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def compute_auto_exposure(arr: np.ndarray, method: str = "logmean", key: float = 0.18, clip_ev: float = 4.0) -> float:
    """
    Compute exposure adjustment in EV such that scene luminance maps to `key`.
    Methods:
      - 'logmean' : log-average luminance (Reinhard-style)
      - 'median'  : median luminance based
    Returns exposure in EV (float). Positive -> brighten, Negative -> darken.
    """
    L = luminance(arr)
    # avoid zero values
    eps = 1e-6
    if method == "median":
        Lm = float(np.median(L.flatten()))
    else:  # logmean
        Lm = float(np.exp(np.mean(np.log(np.clip(L, eps, None)))))

    # compute exposure EV: key = Lm * 2^EV => EV = log2(key / Lm)
    # If image is HDR Lm may be > key and EV negative.
    if Lm <= 0 or np.isnan(Lm):
        return 0.0
    ev = np.log2(key / Lm)
    # clamp
    ev = float(max(-clip_ev, min(clip_ev, ev)))
    return ev


# ------------------------------
# Tone mapping / processing
# ------------------------------
def process_image(arr: np.ndarray,
                  tone: str = "agx",
                  ocio_config: str = None,
                  exposure_ev: float = 0.0,
                  contrast: float = 1.05,
                  saturation: float = 1.03) -> np.ndarray:
    """Apply tone mapping using AgX or Hable. `arr` must be scene-linear float32."""
    if tone.startswith("agx") and HAVE_OCIO:
        try:
            variant = "base"
            if "-" in tone:
                _, variant = tone.split("-", 1)

            display, view = guess_agx_view(ocio_config)
            available = list_ocio_views(ocio_config)

            for disp, views in available.items():
                for candidate in [f"AgX - {variant.capitalize()} Contrast", f"AgX {variant.capitalize()} Contrast"]:
                    if candidate in views:
                        display, view = disp, candidate
                        break

            arr = arr * (2.0 ** exposure_ev)
            log.info(f"  AgX: {display} / {view}, Exposure {exposure_ev:+.2f} EV")
            arr = apply_agx_ocio(arr, ocio_config, "Utility - Linear - sRGB", display, view, encode_srgb=True)
        except Exception as e:
            log.warning(f"AgX failed ({e}), falling back to Hable.")
            tone = "hable"

    if tone == "hable" or not HAVE_OCIO:
        exposure = 2.0 ** exposure_ev
        arr = apply_filmic_hable(arr, exposure=exposure, white_point=11.2,
                                 desat_highlights=True, encode_srgb=True)
        log.info(f"  Hable Filmic applied, Exposure {exposure_ev:+.2f} EV")

    # Post-tone adjustments (display space)
    if contrast != 1.0:
        arr = (arr - 0.5) * contrast + 0.5
        arr = np.clip(arr, 0.0, 1.0)
        log.info(f"  Contrast {contrast:.2f}x")

    if saturation != 1.0:
        gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        arr = gray[..., None] + (arr - gray[..., None]) * saturation
        arr = np.clip(arr, 0.0, 1.0)
        log.info(f"  Saturation {saturation:.2f}x")

    return arr


# ------------------------------
# Single-file processing worker
# ------------------------------
def process_single_file(img_path: Path, out_path: Path, *,
                        tone: str, ocio_config: str, base_exposure: float,
                        contrast: float, saturation: float, auto_exposure: str,
                        key: float, save_linear: bool, quality: int):
    """
    Worker: load image, compute auto-exposure (if requested), tone map, save.
    Returns tuple (img_path, success_bool, message)
    """
    try:
        # Load in scene-linear domain
        arr = load_image(img_path, to_linear=True)

        # Auto exposure if requested
        exposure_ev = base_exposure
        if auto_exposure and auto_exposure.lower() != "none":
            computed_ev = compute_auto_exposure(arr, method=auto_exposure.lower(), key=key)
            # combine base + auto
            exposure_ev = float(base_exposure) + computed_ev
            log.info(f"{img_path.name}: auto-exposure {computed_ev:+.2f} EV -> total {exposure_ev:+.2f} EV")

        # Process and get display-space array
        out_arr = process_image(arr, tone=tone, ocio_config=ocio_config,
                                exposure_ev=exposure_ev, contrast=contrast,
                                saturation=saturation)

        # Save
        save_image(out_arr, out_path, quality=quality, save_linear=save_linear)
        return (img_path, True, "OK")
    except Exception as e:
        return (img_path, False, str(e))


# ------------------------------
# Batch folder processor with threading
# ------------------------------
def process_folder(input_dir: Path,
                   output_dir: Path,
                   *,
                   tone: str = "agx",
                   ocio_config: str = None,
                   base_exposure: float = 0.0,
                   contrast: float = 1.05,
                   saturation: float = 1.03,
                   auto_exposure: str = "logmean",
                   key: float = 0.18,
                   save_linear: bool = False,
                   quality: int = 95,
                   workers: int = 4):
    """Batch process an entire folder with ThreadPoolExecutor."""
    image_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".exr", ".hdr")
    images = sorted([f for f in input_dir.glob("*") if f.suffix.lower() in image_exts])
    if not images:
        log.warning(f"No images found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"\nProcessing {len(images)} images from {input_dir} with {workers} workers...\n")

    tasks = []
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for img_path in images:
            out_path = output_dir / img_path.name
            fut = ex.submit(process_single_file, img_path, out_path,
                            tone=tone, ocio_config=ocio_config, base_exposure=base_exposure,
                            contrast=contrast, saturation=saturation, auto_exposure=auto_exposure,
                            key=key, save_linear=save_linear, quality=quality)
            futures[fut] = img_path

        # progress bar
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Enhancing images", unit="img"):
            img_path = futures[fut]
            try:
                img_path, ok, msg = fut.result()
                if not ok:
                    log.error(f"⚠️ Error processing {img_path.name}: {msg}")
                else:
                    # quiet on success; save_image already logs saved file
                    pass
            except Exception as e:
                log.error(f"⚠️ Fatal error for {img_path.name}: {e}")

    log.info("\n✅ Batch complete.\n")


# ------------------------------
# Main CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="AgX Batch Tone Mapper for Architectural Renders (multithreaded)")
    parser.add_argument("--input", type=Path, help="Input image file")
    parser.add_argument("--output", type=Path, help="Output image file")
    parser.add_argument("--input-dir", type=Path, help="Input folder (batch mode)")
    parser.add_argument("--output-dir", type=Path, help="Output folder (batch mode)")
    parser.add_argument("--tone", choices=["agx", "agx-base", "agx-medium", "agx-high", "hable"],
                        default="agx", help="Tone mapping method")
    parser.add_argument("--ocio-config", type=str, default=None, help="Path to OCIO config")
    parser.add_argument("--exposure", type=float, default=0.0, help="Base exposure in EV stops (added to auto-exposure)")
    parser.add_argument("--contrast", type=float, default=1.05, help="Contrast multiplier")
    parser.add_argument("--saturation", type=float, default=1.03, help="Saturation multiplier")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--save-linear", action="store_true", help="Save as EXR in linear space")
    parser.add_argument("--archviz", action="store_true", help="Use calibrated exposure/contrast defaults for architecture")
    parser.add_argument("--list-views", action="store_true", help="List available OCIO views")
    parser.add_argument("--auto-exposure", choices=["logmean", "median", "none"], default="logmean",
                        help="Per-image auto-exposure method")
    parser.add_argument("--key", type=float, default=0.18, help="Target key luminance for auto-exposure (default 0.18)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))), help="Number of worker threads")

    args = parser.parse_args()

    if args.list_views:
        if not HAVE_OCIO:
            log.error("PyOpenColorIO not installed. Install with: pip install opencolorio")
            return
        cfg = args.ocio_config or os.environ.get("OCIO")
        views = list_ocio_views(cfg)
        log.info("\nAvailable OCIO Displays and Views:\n" + "="*50)
        for display, vlist in views.items():
            log.info(f"\nDisplay: {display}")
            for v in vlist:
                log.info(f"  • {v}")
        return

    # Apply architectural preset
    if args.archviz:
        log.info("Applying architectural visualization preset (+0.7 EV, 1.1 contrast, 1.05 saturation)")
        args.exposure += 0.7
        args.contrast = 1.1
        args.saturation = 1.05

    # Single file mode
    if args.input and args.output:
        log.info(f"Processing single: {args.input} → {args.output}")
        arr = load_image(args.input, to_linear=True)
        exposure_ev = args.exposure
        if args.auto_exposure != "none":
            auto_ev = compute_auto_exposure(arr, method=args.auto_exposure, key=args.key)
            exposure_ev = float(exposure_ev + auto_ev)
            log.info(f"Auto-exposure {auto_ev:+.2f} EV -> total {exposure_ev:+.2f} EV")
        out_arr = process_image(arr, tone=args.tone, ocio_config=args.ocio_config,
                                exposure_ev=exposure_ev, contrast=args.contrast,
                                saturation=args.saturation)
        save_image(out_arr, args.output, quality=args.quality, save_linear=args.save_linear)
        return

    # Batch mode
    if args.input_dir and args.output_dir:
        process_folder(args.input_dir, args.output_dir,
                       tone=args.tone, ocio_config=args.ocio_config,
                       base_exposure=args.exposure, contrast=args.contrast,
                       saturation=args.saturation, auto_exposure=args.auto_exposure,
                       key=args.key, save_linear=args.save_linear, quality=args.quality,
                       workers=args.workers)
        return

    parser.print_help()


if __name__ == "__main__":
    main()


