"""Enhanced depth cache with metadata sidecar support.

Provides backward-compatible caching that supports both:
- Legacy format: .npy (depth array only)
- Enhanced format: .npz + .json sidecar (depth + metadata + focal length)

See ADR-019 for architectural rationale.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from .protocol import DepthResult

logger = logging.getLogger(__name__)


class DepthCacheWriter:
    """Write depth results to cache with metadata sidecar.

    Enhanced format preserves:
    - Depth array (float32)
    - Depth units (relative/meters)
    - Focal length (if metric)
    - Field of view (if metric)
    - Backend provenance
    - Timestamp

    Backward compatible: reads legacy .npy caches.
    """

    def __init__(self, cache_dir: Path):
        """Initialize cache writer.

        Args:
            cache_dir: Directory for cache storage.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def write(self, cache_key: str, result: "DepthResult") -> Path:
        """Write depth result to cache.

        Uses enhanced format (.npz + .json) for metric depth,
        legacy format (.npy) for relative depth.

        Args:
            cache_key: Unique cache identifier.
            result: DepthResult to cache.

        Returns:
            Path to primary cache file.
        """
        if result.is_metric:
            return self._write_enhanced(cache_key, result)
        else:
            return self._write_legacy(cache_key, result)

    def _write_enhanced(self, cache_key: str, result: "DepthResult") -> Path:
        """Write enhanced format (.npz + .json sidecar)."""
        npz_path = self.cache_dir / f"{cache_key}.npz"
        json_path = self.cache_dir / f"{cache_key}.json"

        # Build data dict for npz
        npz_data: Dict[str, Any] = {
            "depth": result.depth_map.astype(np.float32),
        }

        # Add optional fields
        if result.focal_length_px is not None:
            npz_data["focal_length_px"] = np.array([result.focal_length_px])
        if result.field_of_view_deg is not None:
            npz_data["fov_deg"] = np.array([result.field_of_view_deg])

        # Write compressed npz
        np.savez_compressed(npz_path, **npz_data)
        logger.debug(f"Wrote enhanced cache: {npz_path}")

        # Build metadata sidecar
        metadata = {
            "cache_version": "2.0",
            "format": "enhanced",
            "depth_units": result.depth_units,
            "backend_id": result.backend_id,
            "device": result.device,
            "dtype": result.dtype,
            "input_size": (list(result.input_size) if result.input_size else None),
            "focal_length_px": result.focal_length_px,
            "field_of_view_deg": result.field_of_view_deg,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "warnings": result.warnings,
            "provenance": result.metadata,
        }

        # Write JSON sidecar
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.debug(f"Wrote metadata sidecar: {json_path}")

        return npz_path

    def _write_legacy(self, cache_key: str, result: "DepthResult") -> Path:
        """Write legacy format (.npy only)."""
        npy_path = self.cache_dir / f"{cache_key}.npy"
        np.save(npy_path, result.depth_map.astype(np.float32))
        logger.debug(f"Wrote legacy cache: {npy_path}")
        return npy_path

    def read(self, cache_key: str) -> Optional["DepthResult"]:
        """Read depth result from cache (backward compatible).

        Tries enhanced format first, falls back to legacy.

        Args:
            cache_key: Cache identifier.

        Returns:
            DepthResult if found, None otherwise.
        """
        # Try enhanced format first
        npz_path = self.cache_dir / f"{cache_key}.npz"
        json_path = self.cache_dir / f"{cache_key}.json"

        if npz_path.exists():
            try:
                return self._read_enhanced(npz_path, json_path)
            except Exception as e:
                logger.warning("Failed to read enhanced" f" cache {npz_path}: {e}")

        # Fallback to legacy format
        npy_path = self.cache_dir / f"{cache_key}.npy"
        if npy_path.exists():
            try:
                return self._read_legacy(npy_path)
            except Exception as e:
                logger.warning(f"Failed to read legacy cache {npy_path}: {e}")

        return None

    def _read_enhanced(self, npz_path: Path, json_path: Path) -> "DepthResult":
        """Read enhanced format (.npz + .json sidecar)."""
        from .protocol import DepthResult

        # Load npz data
        data = np.load(npz_path)
        depth_map = data["depth"]

        # Extract optional fields
        focal_length_px = None
        if "focal_length_px" in data:
            focal_length_px = float(data["focal_length_px"][0])

        fov_deg = None
        if "fov_deg" in data:
            fov_deg = float(data["fov_deg"][0])

        # Load metadata sidecar
        metadata: Dict[str, Any] = {}
        if json_path.exists():
            with open(json_path) as f:
                metadata = json.load(f)

        # Reconstruct DepthResult
        return DepthResult(
            depth_map=depth_map,
            original_image=np.array([]),  # Not cached
            metadata=metadata.get("provenance", {}),
            depth_units=metadata.get("depth_units", "meters"),
            focal_length_px=focal_length_px,
            field_of_view_deg=fov_deg,
            backend_id=metadata.get("backend_id"),
            device=metadata.get("device"),
            dtype=metadata.get("dtype"),
            input_size=tuple(metadata.get("input_size", [])) or None,
            warnings=metadata.get("warnings", []) + ["loaded from cache"],
        )

    def _read_legacy(self, npy_path: Path) -> "DepthResult":
        """Read legacy format (.npy only)."""
        from .protocol import DepthResult

        depth_map = np.load(npy_path)

        return DepthResult(
            depth_map=depth_map,
            original_image=np.array([]),  # Not cached
            metadata={},
            depth_units="relative",  # Legacy is always relative
            warnings=["loaded from legacy cache (no metadata)"],
        )

    def exists(self, cache_key: str) -> bool:
        """Check if cache entry exists (any format)."""
        npz_path = self.cache_dir / f"{cache_key}.npz"
        npy_path = self.cache_dir / f"{cache_key}.npy"
        return npz_path.exists() or npy_path.exists()

    def delete(self, cache_key: str) -> bool:
        """Delete cache entry (all associated files)."""
        deleted = False

        for suffix in [".npz", ".json", ".npy"]:
            path = self.cache_dir / f"{cache_key}{suffix}"
            if path.exists():
                path.unlink()
                deleted = True
                logger.debug(f"Deleted cache file: {path}")

        return deleted
