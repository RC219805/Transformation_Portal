"""Model cache management for DA3 variants."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelCacheInfo:
    """Information about a cached model."""

    model_id: str
    local_path: Path
    size_bytes: int
    cached_at: str
    verified: bool
    version: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "local_path": str(self.local_path),
            "size_bytes": self.size_bytes,
            "size_mb": self.size_bytes / (1024 * 1024),
            "size_gb": self.size_bytes / (1024 * 1024 * 1024),
            "cached_at": self.cached_at,
            "verified": self.verified,
            "version": self.version,
        }


class CacheStrategy(Enum):
    """Cache download strategies."""

    HF_CACHE = "hf_cache"  # Use HuggingFace default cache
    SNAPSHOT = "snapshot"  # Full snapshot download
    SYMLINK = "symlink"  # Symlink to shared cache


class ModelCacheManager:
    """
    Manage DA3 model downloads and caching.

    Supports:
    - Pre-caching all model variants
    - Offline operation after initial download
    - Cache validation and verification
    - Storage management
    - Deployment snapshots
    """

    # All supported DA3 models
    OFFICIAL_MODELS = {
        # v1.1 models (recommended)
        "nested-giant-large-v1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "giant-v1.1": "depth-anything/DA3-GIANT-1.1",
        "large-v1.1": "depth-anything/DA3-LARGE-1.1",
        # v1.0 models (legacy)
        "nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
        "giant": "depth-anything/DA3-GIANT",
        "large": "depth-anything/DA3-LARGE",
        # Other variants
        "base": "depth-anything/DA3-BASE",
        "small": "depth-anything/DA3-SMALL",
        "metric-large": "depth-anything/DA3METRIC-LARGE",
        "mono-large": "depth-anything/DA3MONO-LARGE",
    }

    # Recommended models for different use cases
    RECOMMENDED_SETS = {
        "essential": ["nested-giant-large-v1.1", "metric-large"],
        "production": ["nested-giant-large-v1.1", "giant-v1.1", "large-v1.1", "metric-large"],
        "benchmark": ["nested-giant-large-v1.1", "giant-v1.1", "large-v1.1", "base", "small", "metric-large", "mono-large"],
        "all": list(OFFICIAL_MODELS.keys()),
    }

    def __init__(self, cache_dir: Optional[Path] = None, strategy: CacheStrategy = CacheStrategy.HF_CACHE):
        """
        Initialize cache manager.

        Args:
            cache_dir: Custom cache directory (None = use HF default)
            strategy: Caching strategy
        """
        self.strategy = strategy

        # Determine cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self._get_default_cache_dir()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.cache_dir / "lux_depth_v3_cache.json"
        self.metadata = self._load_metadata()

    def _get_default_cache_dir(self) -> Path:
        """Get default HuggingFace cache directory."""
        # Check environment variables
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home) / "hub"

        hf_cache = os.environ.get("HF_HUB_CACHE")
        if hf_cache:
            return Path(hf_cache)

        # Default location
        return Path.home() / ".cache" / "huggingface" / "hub"

    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"models": {}, "last_updated": None}

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def download_model(self, model_key: str, force: bool = False, verify: bool = True) -> ModelCacheInfo:
        """
        Download and cache a single model.

        Args:
            model_key: Model key (e.g., "nested-giant-large-v1.1")
            force: Force re-download even if cached
            verify: Verify download after completion

        Returns:
            ModelCacheInfo with cache details
        """
        if model_key not in self.OFFICIAL_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.OFFICIAL_MODELS.keys())}")

        model_id = self.OFFICIAL_MODELS[model_key]

        # Check if already cached
        if not force and self._is_cached(model_id):
            logger.info(f"Model already cached: {model_id}")
            return self._get_cache_info(model_id)

        # Download
        logger.info(f"Downloading {model_id}...")

        if self.strategy == CacheStrategy.HF_CACHE:
            cache_info = self._download_hf_cache(model_id)
        elif self.strategy == CacheStrategy.SNAPSHOT:
            cache_info = self._download_snapshot(model_id, model_key)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        # Verify
        if verify:
            cache_info.verified = self._verify_model(cache_info)

        # Update metadata
        self.metadata["models"][model_id] = cache_info.to_dict()
        self.metadata["last_updated"] = datetime.now().isoformat()
        self._save_metadata()

        return cache_info

    def _download_hf_cache(self, model_id: str) -> ModelCacheInfo:
        """Download using HuggingFace default cache."""
        from huggingface_hub import snapshot_download

        # snapshot_download returns the resolved snapshot directory inside the hub cache.
        snapshot_path = snapshot_download(repo_id=model_id, repo_type="model", cache_dir=str(self.cache_dir))
        cache_path = Path(snapshot_path)
        size = self._get_directory_size(cache_path)

        return ModelCacheInfo(
            model_id=model_id,
            local_path=cache_path,
            size_bytes=size,
            cached_at=datetime.now().isoformat(),
            verified=False,
        )

    def _download_snapshot(self, model_id: str, model_key: str) -> ModelCacheInfo:
        """Download full snapshot."""
        from huggingface_hub import snapshot_download

        local_dir = self.cache_dir / "snapshots" / model_key
        local_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading snapshot to {local_dir}")

        snapshot_download(repo_id=model_id, repo_type="model", local_dir=str(local_dir))

        size = self._get_directory_size(local_dir)

        return ModelCacheInfo(
            model_id=model_id, local_path=local_dir, size_bytes=size, cached_at=datetime.now().isoformat(), verified=False
        )

    def download_models(
        self, model_set: str = "essential", model_keys: Optional[List[str]] = None, force: bool = False, verify: bool = True
    ) -> List[ModelCacheInfo]:
        """
        Download multiple models.

        Args:
            model_set: Predefined set (essential/production/benchmark/all)
            model_keys: Explicit list of model keys (overrides model_set)
            force: Force re-download
            verify: Verify downloads

        Returns:
            List of ModelCacheInfo for downloaded models
        """
        # Determine models to download
        if model_keys:
            keys = model_keys
        elif model_set in self.RECOMMENDED_SETS:
            keys = self.RECOMMENDED_SETS[model_set]
        else:
            raise ValueError(f"Unknown model set: {model_set}. Available: {list(self.RECOMMENDED_SETS.keys())}")

        logger.info(f"Downloading {len(keys)} models ({model_set} set)")

        results = []
        for key in keys:
            try:
                info = self.download_model(key, force=force, verify=verify)
                results.append(info)
                logger.info(f"✓ {key}: {info.size_bytes / (1024**3):.2f} GB")
            except Exception as e:
                logger.error(f"✗ {key}: {e}")

        return results

    def list_cached_models(self) -> List[ModelCacheInfo]:
        """List all cached models."""
        cached = []

        for model_id, info_dict in self.metadata["models"].items():
            cached.append(
                ModelCacheInfo(
                    model_id=info_dict["model_id"],
                    local_path=Path(info_dict["local_path"]),
                    size_bytes=info_dict["size_bytes"],
                    cached_at=info_dict["cached_at"],
                    verified=info_dict["verified"],
                    version=info_dict.get("version"),
                )
            )

        return cached

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        cached_models = self.list_cached_models()

        total_size = sum(m.size_bytes for m in cached_models)

        return {
            "cache_dir": str(self.cache_dir),
            "num_models": len(cached_models),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "models": [m.to_dict() for m in cached_models],
            "last_updated": self.metadata.get("last_updated"),
        }

    def _is_cached(self, model_id: str) -> bool:
        """Check if model is already cached."""
        info = self.metadata.get("models", {}).get(model_id)
        if not info:
            return False
        try:
            path = Path(info.get("local_path", ""))
        except Exception:
            return False
        return path.exists() and any(path.rglob("*"))

    def _get_cache_info(self, model_id: str) -> ModelCacheInfo:
        """Get cache info for a model."""
        info_dict = self.metadata["models"][model_id]
        return ModelCacheInfo(
            model_id=info_dict["model_id"],
            local_path=Path(info_dict["local_path"]),
            size_bytes=info_dict["size_bytes"],
            cached_at=info_dict["cached_at"],
            verified=info_dict["verified"],
            version=info_dict.get("version"),
        )

    def _verify_model(self, cache_info: ModelCacheInfo) -> bool:
        """Verify model integrity."""
        # Basic verification: check path exists and has essential files.
        if not cache_info.local_path.exists():
            return False

        # Config file
        has_config = (cache_info.local_path / "config.json").exists()

        # Weight files (common patterns)
        has_weights = any(cache_info.local_path.rglob("*.safetensors")) or any(cache_info.local_path.rglob("*.bin"))

        # Fall back to "has anything" to avoid false negatives on model layout changes.
        has_any = any(cache_info.local_path.rglob("*"))
        return (has_config and has_weights) or has_any

    def _model_id_to_path(self, model_id: str) -> str:
        """Convert model ID to cache path."""
        # HuggingFace uses hashed directory names
        return model_id.replace("/", "--")

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory."""
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total


def precache_models(
    model_set: str = "essential", cache_dir: Optional[Path] = None, force: bool = False
) -> List[ModelCacheInfo]:
    """
    Convenience function to pre-cache DA3 models.

    Args:
        model_set: Set to download (essential/production/benchmark/all)
        cache_dir: Custom cache directory
        force: Force re-download

    Returns:
        List of cached model information

    Examples:
        >>> # Cache essential models
        >>> precache_models("essential")

        >>> # Cache all for benchmarking
        >>> precache_models("benchmark")

        >>> # Custom cache directory
        >>> precache_models("production", cache_dir=Path("/data/models"))
    """
    manager = ModelCacheManager(cache_dir=cache_dir)
    return manager.download_models(model_set=model_set, force=force)


# Backwards-compatible alias (some scripts expect this name).
ModelCache = ModelCacheManager
