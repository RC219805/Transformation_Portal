"""
Storage Manager for Intelligent Tiered Storage.

Manages internal SSD + external T9 storage tiering to optimize
disk space and I/O performance. Automatically migrates large files
to external storage while maintaining transparent access via symlinks.

Key Features:
- Intelligent tier selection (internal vs T9)
- Auto-migration of large files (>2GB)
- Space management and pre-flight checks
- Symlink management for backward compatibility
- Graceful degradation when T9 unavailable

Performance Target: Eliminate disk space bottlenecks, enable 100+ image batches
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict
import psutil

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for tiered storage management."""
    
    # Storage tiers
    internal_ssd_path: str = "."
    external_t9_path: Optional[str] = None
    
    # Auto-migration settings
    auto_migrate_threshold_gb: float = 2.0  # Files >2GB migrate to T9
    prefer_external_for_upscaled: bool = True  # Upscaled outputs → T9
    
    # Space management
    min_free_space_gb: float = 10.0  # Minimum free space on internal
    warning_threshold_percent: float = 80.0  # Warn at 80% usage
    critical_threshold_percent: float = 90.0  # Critical at 90% usage
    
    # Symlink management
    create_symlinks: bool = True  # Maintain backward compatibility
    symlink_base_dir: Optional[str] = None  # Base dir for symlinks (default: internal)
    
    # Cleanup
    auto_cleanup_enabled: bool = True
    cleanup_on_critical: bool = True


@dataclass
class StorageStats:
    """Statistics for storage tier."""
    
    path: Path
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    percent_used: float = 0.0
    available: bool = True
    
    @property
    def has_space(self) -> bool:
        """Check if tier has sufficient space."""
        return self.available and self.free_gb > 5.0


class StorageManager:
    """
    Intelligent storage tier manager for internal SSD + external T9.
    
    Strategy:
    1. Small files (<2GB): Internal SSD for speed
    2. Large files (>2GB): T9 for capacity
    3. Hot data: Keep on internal for performance
    4. Cold data: Migrate to T9 after processing
    
    Usage:
        manager = StorageManager(config)
        
        # Get optimal write path
        output_path = manager.get_optimal_write_path('upscaled', size_gb=3.5)
        
        # Auto-migrate after write
        manager.auto_migrate_if_needed(output_path)
        
        # Check space before batch
        if not manager.ensure_space_available(required_gb=50):
            raise RuntimeError("Insufficient disk space")
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize storage manager.
        
        Args:
            config: Storage configuration
        """
        self.config = config
        
        # Initialize storage paths
        self.internal = Path(config.internal_ssd_path).resolve()
        self.t9 = Path(config.external_t9_path).resolve() if config.external_t9_path else None
        
        # Symlink base directory
        if config.symlink_base_dir:
            self.symlink_base = Path(config.symlink_base_dir).resolve()
        else:
            self.symlink_base = self.internal
        
        # Validate paths
        if not self.internal.exists():
            self.internal.mkdir(parents=True, exist_ok=True)
        
        if self.t9 and not self.t9.exists():
            logger.warning(f"T9 path not found: {self.t9} - will operate in degraded mode")
            self.t9 = None
        
        # Migration statistics
        self.stats = {
            'files_migrated': 0,
            'bytes_migrated': 0,
            'symlinks_created': 0,
        }
        
        logger.info(
            f"StorageManager initialized: internal={self.internal}, "
            f"t9={self.t9}, auto_migrate={config.auto_migrate_threshold_gb}GB"
        )
    
    def get_storage_stats(self, tier: str = 'internal') -> StorageStats:
        """
        Get storage statistics for tier.
        
        Args:
            tier: 'internal' or 't9'
        
        Returns:
            StorageStats object
        """
        if tier == 't9' and not self.t9:
            return StorageStats(path=Path('/dev/null'), available=False)
        
        path = self.internal if tier == 'internal' else self.t9
        
        try:
            usage = psutil.disk_usage(str(path))
            
            return StorageStats(
                path=path,
                total_gb=usage.total / 1e9,
                used_gb=usage.used / 1e9,
                free_gb=usage.free / 1e9,
                percent_used=usage.percent,
                available=True,
            )
        except Exception as e:
            logger.error(f"Failed to get stats for {tier}: {e}")
            return StorageStats(path=path, available=False)
    
    def get_optimal_write_path(
        self,
        file_type: str,
        estimated_size_gb: float,
        prefer_tier: Optional[str] = None
    ) -> Path:
        """
        Determine optimal storage tier for write.
        
        Strategy:
        - Large files (>threshold): T9 if available
        - Small files: Internal SSD
        - Manual override via prefer_tier
        
        Args:
            file_type: File type category ('upscaled', 'depth', 'graded', etc.)
            estimated_size_gb: Expected file size
            prefer_tier: Optional tier preference ('internal' | 't9')
        
        Returns:
            Path object for write location
        """
        # Manual override
        if prefer_tier == 't9' and self.t9:
            return self.t9 / file_type
        elif prefer_tier == 'internal':
            return self.internal / file_type
        
        # Auto-migration threshold
        if estimated_size_gb >= self.config.auto_migrate_threshold_gb and self.t9:
            # Check T9 has space
            t9_stats = self.get_storage_stats('t9')
            if t9_stats.has_space:
                logger.debug(
                    f"Large file ({estimated_size_gb:.2f}GB) → T9: {file_type}"
                )
                return self.t9 / file_type
        
        # Default: internal SSD
        return self.internal / file_type
    
    def auto_migrate_if_needed(self, file_path: Path) -> Optional[Path]:
        """
        Automatically migrate file to T9 if above threshold.
        
        Creates symlink at original location for backward compatibility.
        
        Args:
            file_path: File to potentially migrate
        
        Returns:
            New path if migrated, None if not migrated
        """
        if not self.t9:
            return None
        
        # Check file size
        try:
            size_gb = file_path.stat().st_size / 1e9
        except FileNotFoundError:
            logger.warning(f"File not found for migration: {file_path}")
            return None
        
        # Check threshold
        if size_gb < self.config.auto_migrate_threshold_gb:
            logger.debug(f"File too small for migration ({size_gb:.2f}GB): {file_path.name}")
            return None
        
        # Check if already on T9
        if self.t9 in file_path.parents:
            logger.debug(f"File already on T9: {file_path.name}")
            return None
        
        # Migrate to T9
        return self._migrate_to_t9(file_path)
    
    def _migrate_to_t9(self, source_path: Path) -> Path:
        """
        Migrate file from internal to T9.
        
        Args:
            source_path: Source file on internal
        
        Returns:
            Destination path on T9
        """
        if not self.t9:
            raise RuntimeError("T9 not available for migration")
        
        # Calculate relative path from internal base
        try:
            rel_path = source_path.relative_to(self.internal)
        except ValueError:
            # If not under internal, use just filename
            rel_path = source_path.name
        
        # Destination on T9
        dest_path = self.t9 / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Migrating {source_path.name} to T9 ({source_path.stat().st_size/1e9:.2f}GB)...")
        
        # Copy file
        start_time = __import__('time').time()
        shutil.copy2(source_path, dest_path)
        copy_time = __import__('time').time() - start_time
        
        # Verify copy
        if not dest_path.exists() or dest_path.stat().st_size != source_path.stat().st_size:
            raise RuntimeError(f"Migration verification failed: {source_path.name}")
        
        logger.info(f"Copy complete: {copy_time:.1f}s")
        
        # Create symlink
        if self.config.create_symlinks:
            # Remove original file
            source_path.unlink()
            
            # Create symlink
            source_path.symlink_to(dest_path)
            self.stats['symlinks_created'] += 1
            
            logger.info(f"Symlink created: {source_path} → {dest_path}")
        else:
            # Just remove original
            source_path.unlink()
            logger.info(f"Original file removed (no symlink)")
        
        # Update stats
        self.stats['files_migrated'] += 1
        self.stats['bytes_migrated'] += dest_path.stat().st_size
        
        return dest_path
    
    def ensure_space_available(self, required_gb: float, tier: str = 'internal') -> bool:
        """
        Pre-flight check: ensure sufficient space available.
        
        Args:
            required_gb: Required free space
            tier: Storage tier to check
        
        Returns:
            True if sufficient space, False otherwise
        """
        stats = self.get_storage_stats(tier)
        
        if not stats.available:
            logger.error(f"Tier {tier} not available")
            return False
        
        if stats.free_gb < required_gb:
            logger.error(
                f"Insufficient space on {tier}: "
                f"free={stats.free_gb:.1f}GB, required={required_gb:.1f}GB"
            )
            
            # Try cleanup if enabled
            if self.config.auto_cleanup_enabled:
                logger.info("Attempting auto-cleanup...")
                self._auto_cleanup(tier, target_free_gb=required_gb)
                
                # Re-check after cleanup
                stats = self.get_storage_stats(tier)
                if stats.free_gb >= required_gb:
                    logger.info(f"Cleanup successful, space available: {stats.free_gb:.1f}GB")
                    return True
            
            return False
        
        # Check warning thresholds
        if stats.percent_used >= self.config.critical_threshold_percent:
            logger.warning(
                f"CRITICAL: {tier} usage at {stats.percent_used:.1f}% "
                f"(free: {stats.free_gb:.1f}GB)"
            )
            
            if self.config.cleanup_on_critical:
                self._auto_cleanup(tier)
        
        elif stats.percent_used >= self.config.warning_threshold_percent:
            logger.warning(
                f"WARNING: {tier} usage at {stats.percent_used:.1f}% "
                f"(free: {stats.free_gb:.1f}GB)"
            )
        
        return True
    
    def _auto_cleanup(self, tier: str, target_free_gb: Optional[float] = None):
        """
        Automatic cleanup of old files.
        
        Strategy:
        1. Remove cache files (.cache, temp directories)
        2. Migrate large files to T9 if available
        3. Remove old checkpoints
        
        Args:
            tier: Tier to clean ('internal' | 't9')
            target_free_gb: Optional target free space
        """
        logger.info(f"Auto-cleanup started for {tier}")
        
        path = self.internal if tier == 'internal' else self.t9
        if not path:
            return
        
        freed_gb = 0.0
        
        # 1. Clean cache directories
        cache_dirs = ['.cache', '__pycache__', '.pytest_cache']
        for cache_name in cache_dirs:
            for cache_dir in path.rglob(cache_name):
                if cache_dir.is_dir():
                    try:
                        dir_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                        shutil.rmtree(cache_dir)
                        freed_gb += dir_size / 1e9
                        logger.info(f"Removed cache: {cache_dir} ({dir_size/1e9:.2f}GB)")
                    except Exception as e:
                        logger.warning(f"Failed to remove {cache_dir}: {e}")
        
        # 2. Clean old checkpoints (keep last 3)
        checkpoint_dirs = list(path.glob('checkpoints_*'))
        if len(checkpoint_dirs) > 3:
            checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime)
            for old_checkpoint in checkpoint_dirs[:-3]:
                try:
                    dir_size = sum(f.stat().st_size for f in old_checkpoint.rglob('*') if f.is_file())
                    shutil.rmtree(old_checkpoint)
                    freed_gb += dir_size / 1e9
                    logger.info(f"Removed old checkpoint: {old_checkpoint} ({dir_size/1e9:.2f}GB)")
                except Exception as e:
                    logger.warning(f"Failed to remove {old_checkpoint}: {e}")
        
        # 3. Migrate large files to T9 (if internal and T9 available)
        if tier == 'internal' and self.t9:
            for output_dir in path.glob('output_*'):
                if not output_dir.is_dir():
                    continue
                
                for file in output_dir.rglob('*.tif'):
                    try:
                        size_gb = file.stat().st_size / 1e9
                        if size_gb >= self.config.auto_migrate_threshold_gb:
                            self._migrate_to_t9(file)
                            freed_gb += size_gb
                    except Exception as e:
                        logger.warning(f"Failed to migrate {file}: {e}")
        
        logger.info(f"Auto-cleanup complete: freed {freed_gb:.2f}GB")
    
    def get_summary(self) -> Dict:
        """
        Get storage manager summary.
        
        Returns:
            Dictionary with storage statistics and migration info
        """
        internal_stats = self.get_storage_stats('internal')
        t9_stats = self.get_storage_stats('t9')
        
        return {
            'internal': {
                'path': str(internal_stats.path),
                'total_gb': internal_stats.total_gb,
                'used_gb': internal_stats.used_gb,
                'free_gb': internal_stats.free_gb,
                'percent_used': internal_stats.percent_used,
            },
            't9': {
                'available': t9_stats.available,
                'path': str(t9_stats.path) if t9_stats.available else None,
                'total_gb': t9_stats.total_gb if t9_stats.available else 0,
                'used_gb': t9_stats.used_gb if t9_stats.available else 0,
                'free_gb': t9_stats.free_gb if t9_stats.available else 0,
                'percent_used': t9_stats.percent_used if t9_stats.available else 0,
            },
            'migration': {
                'files_migrated': self.stats['files_migrated'],
                'bytes_migrated_gb': self.stats['bytes_migrated'] / 1e9,
                'symlinks_created': self.stats['symlinks_created'],
            },
        }


# Convenience functions

def create_storage_manager(
    internal_path: str = ".",
    t9_path: Optional[str] = None,
    auto_migrate_gb: float = 2.0
) -> StorageManager:
    """
    Create storage manager with sensible defaults.
    
    Args:
        internal_path: Internal SSD path
        t9_path: Optional T9 external path
        auto_migrate_gb: File size threshold for auto-migration
    
    Returns:
        StorageManager instance
    """
    config = StorageConfig(
        internal_ssd_path=internal_path,
        external_t9_path=t9_path,
        auto_migrate_threshold_gb=auto_migrate_gb,
    )
    return StorageManager(config)
