"""Depth map caching for AUTO depth generation."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class DepthCacheManager:
    """
    Cache file: <cache_key>.npz
      - depth: float32 (H,W) [0,1]
      - metadata: dict (stored as numpy object)
      - confidence_proxy: float

    Design goals:
      - Deterministic cache keys
      - Avoid hashing entire multi-GB TIFFs (hash metadata + boundary samples)
    """

    def __init__(self, cache_dir: Path, logger=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.hits = 0
        self.misses = 0

    def _fast_file_fingerprint(self, p: Path, sample_bytes: int = 1024 * 1024) -> str:
        """
        Fingerprint using:
          - size, mtime_ns
          - first N bytes
          - last N bytes
        """
        h = hashlib.sha256()
        st = p.stat()
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
        with p.open("rb") as f:
            head = f.read(sample_bytes)
            h.update(head)
            if st.st_size > sample_bytes:
                f.seek(max(0, st.st_size - sample_bytes))
                tail = f.read(sample_bytes)
                h.update(tail)
        return h.hexdigest()[:16]

    def compute_cache_key(self, img_path: Path, model_name: str, params_fingerprint: str) -> str:
        """Generate deterministic cache key from image + model + config."""
        try:
            input_fp = self._fast_file_fingerprint(img_path)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Depth cache fingerprint failed for {img_path}: {e}")
            input_fp = "unknown"

        stem = img_path.stem
        model_tag = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
        cfg_tag = (params_fingerprint or "default")[:10]
        return f"{stem}_{input_fp}_{model_tag}_{cfg_tag}"

    def _path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.npz"

    def is_cached(self, cache_key: str) -> bool:
        return self._path(cache_key).exists()

    def load(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached depth map + metadata."""
        p = self._path(cache_key)
        try:
            data = np.load(p, allow_pickle=True)
            self.hits += 1
            return {
                "depth": data["depth"],
                "metadata": data["metadata"].item(),
                "confidence_proxy": float(data["confidence_proxy"]),
            }
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Depth cache load failed for {cache_key}: {e}")
            return None

    def save(
        self,
        cache_key: str,
        depth: np.ndarray,
        metadata: dict,
        confidence_proxy: float,
    ) -> None:
        """Save depth map + metadata to cache."""
        p = self._path(cache_key)
        try:
            np.savez_compressed(
                p,
                depth=depth.astype(np.float32),
                metadata=np.array(metadata, dtype=object),
                confidence_proxy=np.array(float(confidence_proxy)),
            )
            self.misses += 1
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Depth cache save failed for {cache_key}: {e}")

    def stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "cache_dir": str(self.cache_dir),
        }
