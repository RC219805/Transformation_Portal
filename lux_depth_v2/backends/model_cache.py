# lux_depth_v2/backends/model_cache.py
"""
Secure, atomic model download and caching for EfficientSAM ONNX models.

Features:
- stdlib-only (no requests dependency)
- SHA256 verification (optional but recommended)
- atomic file writes (temp + rename)
- network operation logging
- offline-by-default

Notes:
- This module NEVER downloads automatically unless explicitly requested.
- Default CI behavior: no network, model availability check only.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

log = logging.getLogger(__name__)


class ModelDownloadError(RuntimeError):
    """Raised when model download or verification fails."""


# Known EfficientSAM models (verified URLs and SHA256 checksums)
DEFAULT_MODELS = {
    "efficientsam_s": {
        "url": "https://huggingface.co/yunyangx/EfficientSAM/resolve/main/efficientsam_s.onnx",
        "sha256": "b257787eeecdfd0db0626f83a8241874c35c74eb4c25c4d12ff0a478f90f30f9",
        "size_mb": 101,
    },
    "efficientsam_ti_vit_s": {
        "url": "https://huggingface.co/yunyangx/efficientvit-sam/resolve/main/efficientsam_ti_s_encoder.onnx",
        "sha256": None,  # TODO: add verified SHA256 after first download
        "size_mb": 40,
    },
    "efficientsam_ti_vit_b": {
        "url": "https://huggingface.co/yunyangx/efficientvit-sam/resolve/main/efficientsam_ti_b_encoder.onnx",
        "sha256": None,
        "size_mb": 140,
    },
}


def compute_sha256(path: Path) -> str:
    """Compute SHA256 of file at path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    *,
    verify_sha256: Optional[str] = None,
    timeout: int = 300,
) -> None:
    """
    Download file from url to dest with optional SHA256 verification.

    Uses atomic write (temp file + rename).

    Parameters
    ----------
    url : str
        URL to download from.
    dest : Path
        Destination path.
    verify_sha256 : Optional[str]
        Expected SHA256 hex digest. If provided, download is verified and
        rejected if hash mismatch.
    timeout : int
        Socket timeout in seconds (default: 300).

    Raises
    ------
    ModelDownloadError
        If download fails or SHA256 verification fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=dest.parent,
        prefix=f".{dest.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_path_str)

    try:
        log.info("Downloading %s → %s", url, dest)
        with urlopen(url, timeout=timeout) as response:
            with open(temp_fd, "wb") as f:
                shutil.copyfileobj(response, f)

        log.info("Download complete: %s", dest)

        if verify_sha256:
            actual = compute_sha256(temp_path)
            if actual != verify_sha256:
                raise ModelDownloadError(
                    f"SHA256 mismatch for {dest.name}: "
                    f"expected {verify_sha256}, got {actual}"
                )
            log.info("SHA256 verified: %s", verify_sha256[:16])
        else:
            actual = compute_sha256(temp_path)
            log.warning(
                "SHA256 verification disabled. Computed hash: %s (store this for future use)",
                actual,
            )

        # Atomic rename
        temp_path.replace(dest)
        log.info("Model cached at %s", dest)

    except (URLError, HTTPError) as exc:
        temp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"Download failed for {url}: {exc}") from exc
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"Download failed for {url}: {exc}") from exc


def get_model_path(
    model_name: str,
    *,
    cache_dir: Optional[Path] = None,
    auto_download: bool = False,
    url_override: Optional[str] = None,
    sha256_override: Optional[str] = None,
) -> Path:
    """
    Get path to cached EfficientSAM model, optionally downloading if missing.

    Parameters
    ----------
    model_name : str
        Logical model name (e.g., 'efficientsam_ti_vit_s').
    cache_dir : Optional[Path]
        Cache directory. Default: weights/efficientsam/
    auto_download : bool
        If True and model missing, attempt download.
        If False (default), raise if missing.
    url_override : Optional[str]
        Override default URL for this model.
    sha256_override : Optional[str]
        Override or provide SHA256 for verification.

    Returns
    -------
    Path
        Path to the cached ONNX file.

    Raises
    ------
    ModelDownloadError
        If model is missing and auto_download is False, or download fails.
    """
    if cache_dir is None:
        cache_dir = Path("weights") / "efficientsam"

    model_path = cache_dir / f"{model_name}.onnx"

    if model_path.exists():
        log.debug("Model found in cache: %s", model_path)
        return model_path

    if not auto_download:
        raise ModelDownloadError(
            f"Model {model_name} not found at {model_path} and auto_download=False"
        )

    # Attempt download
    if url_override:
        url = url_override
        sha256 = sha256_override
    elif model_name in DEFAULT_MODELS:
        info = DEFAULT_MODELS[model_name]
        url = info["url"]
        sha256 = sha256_override or info.get("sha256")
    else:
        raise ModelDownloadError(
            f"Unknown model {model_name} and no URL override provided"
        )

    log.info("Model %s not cached; downloading from %s", model_name, url)
    download_file(url, model_path, verify_sha256=sha256)

    return model_path


def check_model_available(
    model_name: str,
    *,
    cache_dir: Optional[Path] = None,
) -> bool:
    """
    Check if model is available in cache (no download attempt).

    Returns
    -------
    bool
        True if model exists in cache.
    """
    if cache_dir is None:
        cache_dir = Path("weights") / "efficientsam"

    model_path = cache_dir / f"{model_name}.onnx"
    return model_path.exists()
