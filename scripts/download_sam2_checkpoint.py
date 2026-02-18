#!/usr/bin/env python3
"""Download SAM2 model checkpoints from Facebook Research."""

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_URLS = {
    "base": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}


def download_checkpoint(model_size: str, output_dir: Path) -> Path:
    url = CHECKPOINT_URLS[model_size]
    filename = f"sam2_hiera_{'base_plus' if model_size == 'base' else 'large'}.pt"
    output_path = output_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"✅ Checkpoint already exists: {output_path}")
        return output_path

    logger.info(f"📥 Downloading SAM2 {model_size} (~{'200' if model_size == 'base' else '400'} MB)...")

    try:

        def report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            if block_num % 100 == 0:
                mb = downloaded / (1024 * 1024)
                logger.info(f"   {percent}% ({mb:.1f} MB)")

        urllib.request.urlretrieve(url, output_path, reporthook=report)
        logger.info(f"✅ Downloaded: {output_path}")
        return output_path
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Download failed: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Download SAM2 checkpoints")
    parser.add_argument("--model", choices=["base", "large"], default="large")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    download_checkpoint(args.model, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
