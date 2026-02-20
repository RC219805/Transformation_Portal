#!/usr/bin/env python3
"""Download SAM2 model checkpoints from Facebook Research."""

import argparse
import hashlib
import logging
import re
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_URLS = {
    "base": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}

CHECKPOINT_SHA256 = {
    "base": "2257ced71f65bfccb444f4b5c0f4af95a64c865503748e42edee0e40ab36a312",
    "large": "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b",
}

_SHA256_HEX_RE = re.compile(r"^[a-fA-F0-9]{64}$")


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256_hex(expected_sha256: str) -> str:
    """Normalize and validate SHA-256 digest format."""
    normalized = expected_sha256.strip().lower()
    if not _SHA256_HEX_RE.fullmatch(normalized):
        raise ValueError(f"Invalid SHA256 digest format: {expected_sha256!r}")
    return normalized


def download_checkpoint(model_size: str, output_dir: Path, expected_sha256: str | None = None) -> Path:
    url = CHECKPOINT_URLS[model_size]
    filename = f"sam2_hiera_{'base_plus' if model_size == 'base' else 'large'}.pt"
    output_path = output_dir / filename
    expected = expected_sha256 or CHECKPOINT_SHA256.get(model_size)
    if not expected:
        raise RuntimeError(
            f"Missing expected SHA256 for SAM2 {model_size} checkpoint. " "Pass --sha256 with the trusted digest."
        )
    expected = validate_sha256_hex(expected)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        actual_sha = compute_file_sha256(output_path)
        if actual_sha == expected:
            logger.info(f"✅ Checkpoint already exists and checksum verified: {output_path}")
            return output_path
        logger.warning(
            "Existing checkpoint checksum mismatch for %s. Expected %s, got %s. Re-downloading.",
            output_path,
            expected,
            actual_sha,
        )
        output_path.unlink(missing_ok=True)

    logger.info(f"📥 Downloading SAM2 {model_size} (~{'200' if model_size == 'base' else '400'} MB)...")

    try:

        def report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            if block_num % 100 == 0:
                mb = downloaded / (1024 * 1024)
                logger.info(f"   {percent}% ({mb:.1f} MB)")

        temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        urllib.request.urlretrieve(url, temp_path, reporthook=report)
        actual_sha = compute_file_sha256(temp_path)
        if actual_sha != expected:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {filename}. Expected {expected}, got {actual_sha}.")
        temp_path.replace(output_path)
        logger.info(f"✅ Downloaded: {output_path}")
        return output_path
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Download SAM2 checkpoints")
    parser.add_argument("--model", choices=["base", "large"], default="large")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--sha256",
        type=str,
        help="Optional SHA256 override for selected model (defaults to built-in trusted checksum)",
    )
    args = parser.parse_args()

    download_checkpoint(args.model, args.output_dir, expected_sha256=args.sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
