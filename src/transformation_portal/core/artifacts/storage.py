"""
Artifact storage management.

Provides abstraction for storing pipeline artifacts across
different storage backends (local, external drives, cloud).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional
import shutil
import logging

logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Storage backend types."""
    LOCAL = "local"
    EXTERNAL = "external"
    CLOUD = "cloud"


class ArtifactStorage:
    """
    Artifact storage manager.
    
    Manages storage of pipeline artifacts with support for
    multiple storage backends and automatic migration.
    """
    
    def __init__(
        self,
        primary_path: Path,
        external_path: Optional[Path] = None,
        auto_migrate_threshold_mb: float = 2000.0
    ):
        """
        Initialize artifact storage.
        
        Args:
            primary_path: Primary storage path (fast local storage)
            external_path: External storage path (e.g., external drive)
            auto_migrate_threshold_mb: Auto-migrate files larger than this
        """
        self.primary_path = Path(primary_path)
        self.external_path = Path(external_path) if external_path else None
        self.auto_migrate_threshold_mb = auto_migrate_threshold_mb
        
        # Ensure primary path exists
        self.primary_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure external path exists if configured
        if self.external_path:
            try:
                self.external_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"External storage not available: {e}")
                self.external_path = None
    
    def store(
        self,
        file_path: Path,
        relative_path: str,
        backend: Optional[StorageBackend] = None
    ) -> Path:
        """
        Store artifact file.
        
        Args:
            file_path: Source file path
            relative_path: Relative path within storage
            backend: Storage backend (auto-select if None)
            
        Returns:
            Destination path
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        # Auto-select backend if not specified
        if backend is None:
            backend = self._select_backend(file_path)
        
        # Determine destination
        if backend == StorageBackend.EXTERNAL and self.external_path:
            dest_path = self.external_path / relative_path
        else:
            dest_path = self.primary_path / relative_path
        
        # Create parent directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        try:
            shutil.copy2(file_path, dest_path)
            logger.debug(f"Stored artifact: {relative_path} -> {backend.value}")
            return dest_path
        except Exception as e:
            logger.error(f"Failed to store artifact: {e}")
            raise
    
    def retrieve(self, relative_path: str) -> Optional[Path]:
        """
        Retrieve artifact file.
        
        Searches across all storage backends.
        
        Args:
            relative_path: Relative path within storage
            
        Returns:
            Path to artifact or None if not found
        """
        # Check primary storage
        primary_file = self.primary_path / relative_path
        if primary_file.exists():
            return primary_file
        
        # Check external storage
        if self.external_path:
            external_file = self.external_path / relative_path
            if external_file.exists():
                return external_file
        
        return None
    
    def migrate(self, relative_path: str, target_backend: StorageBackend) -> Optional[Path]:
        """
        Migrate artifact to different backend.
        
        Args:
            relative_path: Relative path within storage
            target_backend: Target storage backend
            
        Returns:
            New path or None if migration failed
        """
        # Find current location
        source_path = self.retrieve(relative_path)
        if source_path is None:
            logger.warning(f"Cannot migrate, artifact not found: {relative_path}")
            return None
        
        # Determine target
        if target_backend == StorageBackend.EXTERNAL and self.external_path:
            target_path = self.external_path / relative_path
        else:
            target_path = self.primary_path / relative_path
        
        # Skip if already at target
        if source_path == target_path:
            return target_path
        
        # Migrate
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(target_path))
            logger.info(f"Migrated artifact: {relative_path} -> {target_backend.value}")
            return target_path
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return None
    
    def _select_backend(self, file_path: Path) -> StorageBackend:
        """
        Auto-select storage backend based on file size.
        
        Args:
            file_path: File to store
            
        Returns:
            Selected backend
        """
        # Get file size
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
        except Exception:
            # Default to primary if cannot get size
            return StorageBackend.LOCAL
        
        # Use external for large files
        if self.external_path and size_mb > self.auto_migrate_threshold_mb:
            return StorageBackend.EXTERNAL
        
        return StorageBackend.LOCAL
    
    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage stats
        """
        stats = {}
        
        # Primary storage
        try:
            total, used, free = shutil.disk_usage(self.primary_path)
            stats["primary"] = {
                "path": str(self.primary_path),
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
            }
        except Exception as e:
            logger.debug(f"Failed to get primary storage stats: {e}")
        
        # External storage
        if self.external_path:
            try:
                total, used, free = shutil.disk_usage(self.external_path)
                stats["external"] = {
                    "path": str(self.external_path),
                    "total_gb": total / (1024**3),
                    "used_gb": used / (1024**3),
                    "free_gb": free / (1024**3),
                }
            except Exception as e:
                logger.debug(f"Failed to get external storage stats: {e}")
        
        return stats
