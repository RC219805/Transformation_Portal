"""Mask Cache Manager for Materials v2.

Features:
- PNG mask storage with JSON metadata
- Content-based cache invalidation (SHA256 hashing)
- Quality audit trail (confidence scores, coverage stats)
- Fast cache lookups and batch cleanup
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .logging_utils import setup_logging


class MaskCacheManager:
    """Manage material mask caching with audit trail.
    
    Features:
    - Hash-based cache invalidation
    - PNG mask storage (8-bit or 16-bit)
    - JSON metadata with confidence metrics
    - Batch cleanup utilities
    
    Args:
        cache_dir: Directory for cache storage
        logger: Optional logger instance
    """
    
    def __init__(self, cache_dir: Path, logger=None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.logger = logger or setup_logging("INFO")
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Mask cache enabled: {self.cache_dir}")
    
    def compute_input_hash(self, image_path: Path) -> str:
        """Compute SHA256 hash of input image file.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Hash string in format "sha256:..."
        """
        hasher = hashlib.sha256()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return f"sha256:{hasher.hexdigest()}"
    
    def compute_input_hash_from_array(self, image: np.ndarray) -> str:
        """Compute SHA256 hash of image array.
        
        Args:
            image: Image array
            
        Returns:
            Hash string in format "sha256:..."
        """
        hasher = hashlib.sha256()
        hasher.update(image.tobytes())
        return f"sha256:{hasher.hexdigest()}"
    
    def get_cache_key(self, task_id: str) -> str:
        """Get cache key for task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Cache key string
        """
        return task_id
    
    def is_cached(self, task_id: str, input_hash: str) -> bool:
        """Check if valid cache exists for task.
        
        Args:
            task_id: Task identifier
            input_hash: Input image hash for validation
            
        Returns:
            True if valid cache exists
        """
        if not self.cache_dir:
            return False
        
        metadata_path = self.cache_dir / f"{task_id}_metadata.json"
        if not metadata_path.exists():
            return False
        
        # Load metadata and check hash
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            cached_hash = metadata.get('input_hash', '')
            if cached_hash != input_hash:
                self.logger.debug(f"Cache invalid (hash mismatch): {task_id}")
                return False
            
            # Check mask files exist
            material_counts = metadata.get('confidence_metrics', {}).get('material_counts', {})
            materials = material_counts.keys()
            
            for material in materials:
                mask_path = self.cache_dir / f"{task_id}_{material}_mask.png"
                if not mask_path.exists():
                    self.logger.debug(f"Cache invalid (missing mask): {task_id}/{material}")
                    return False
            
            return True
        except Exception as e:
            self.logger.warning(f"Cache check failed for {task_id}: {e}")
            return False
    
    def load_masks(self, task_id: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Load cached masks and metadata for task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            (masks dict, metadata dict) tuple
        """
        if not self.cache_dir:
            return {}, {}
        
        metadata_path = self.cache_dir / f"{task_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load mask files
        masks = {}
        material_counts = metadata.get('confidence_metrics', {}).get('material_counts', {})
        materials = material_counts.keys()
        
        for material in materials:
            mask_path = self.cache_dir / f"{task_id}_{material}_mask.png"
            if mask_path.exists():
                try:
                    from PIL import Image
                    mask_img = Image.open(mask_path)
                    mask_array = np.array(mask_img).astype(np.float32)
                    
                    # Normalize based on bit depth
                    if mask_array.max() > 1.0:
                        mask_array = mask_array / 255.0
                    
                    masks[material] = mask_array
                except Exception as e:
                    self.logger.warning(f"Failed to load mask {material}: {e}")
        
        self.logger.debug(f"Loaded {len(masks)} cached masks: {task_id}")
        return masks, metadata
    
    def save_masks(
        self,
        task_id: str,
        masks: Dict[str, np.ndarray],
        confidence_metrics,  # ConfidenceMetrics dataclass
        input_hash: str,
        config: Dict,
        metadata: Optional[Dict] = None
    ):
        """Save masks and metadata to cache.
        
        Args:
            task_id: Task identifier
            masks: Material masks dict {material_type: mask_array}
            confidence_metrics: ConfidenceMetrics instance
            input_hash: Input image hash
            config: Segmentation config dict
            metadata: Additional metadata
        """
        if not self.cache_dir:
            return
        
        # Save mask PNGs
        from PIL import Image
        for material, mask in masks.items():
            mask_path = self.cache_dir / f"{task_id}_{material}_mask.png"
            
            # Convert to 8-bit for storage (sufficient for masks)
            if mask.ndim == 2:
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_uint8, mode='L')
            else:
                # Multi-channel mask
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_uint8)
            
            mask_img.save(mask_path, optimize=True)
        
        # Prepare metadata
        meta_dict = {
            'task_id': task_id,
            'input_hash': input_hash,
            'timestamp': time.time(),
            'segmentation_config': config,
            'confidence_metrics': confidence_metrics.to_dict(),
            'version': '2.0',
        }
        
        # Add additional metadata if provided
        if metadata:
            meta_dict.update(metadata)
        
        # Save metadata JSON
        metadata_path = self.cache_dir / f"{task_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(meta_dict, f, indent=2)
        
        self.logger.debug(f"Saved {len(masks)} masks to cache: {task_id}")
    
    def invalidate(self, task_id: str):
        """Invalidate cache for task (delete files).
        
        Args:
            task_id: Task identifier
        """
        if not self.cache_dir:
            return
        
        # Remove mask files and metadata
        count = 0
        for path in self.cache_dir.glob(f"{task_id}_*"):
            path.unlink()
            count += 1
        
        if count > 0:
            self.logger.debug(f"Invalidated cache: {task_id} ({count} files)")
    
    def clear_all(self):
        """Clear all cached data (use with caution)."""
        if not self.cache_dir:
            return
        
        count = 0
        for path in self.cache_dir.glob("*"):
            if path.is_file():
                path.unlink()
                count += 1
        
        self.logger.info(f"Cleared all cache ({count} files)")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir or not self.cache_dir.exists():
            return {
                'enabled': False,
                'total_entries': 0,
                'total_size_mb': 0.0,
            }
        
        # Count metadata files (one per task)
        metadata_files = list(self.cache_dir.glob("*_metadata.json"))
        total_entries = len(metadata_files)
        
        # Calculate total size
        total_size = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("*")
            if f.is_file()
        )
        total_size_mb = total_size / (1024 * 1024)
        
        # Count by material type
        material_counts = {}
        for mask_file in self.cache_dir.glob("*_mask.png"):
            # Extract material type from filename
            # Format: {task_id}_{material}_mask.png
            parts = mask_file.stem.rsplit('_', 2)
            if len(parts) >= 2:
                material = parts[-2]
                material_counts[material] = material_counts.get(material, 0) + 1
        
        return {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'total_entries': total_entries,
            'total_size_mb': round(total_size_mb, 2),
            'material_counts': material_counts,
        }
    
    def cleanup_old_cache(self, max_age_days: int = 7):
        """Remove cache entries older than specified age.
        
        Args:
            max_age_days: Maximum age in days
        """
        if not self.cache_dir:
            return
        
        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()
        
        removed_count = 0
        for metadata_file in self.cache_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                timestamp = metadata.get('timestamp', 0)
                age = current_time - timestamp
                
                if age > max_age_seconds:
                    # Remove this task's cache
                    task_id = metadata.get('task_id', metadata_file.stem.replace('_metadata', ''))
                    self.invalidate(task_id)
                    removed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to check age for {metadata_file}: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} old cache entries (>{max_age_days} days)")
    
    def validate_cache_integrity(self) -> Dict:
        """Validate cache integrity (check for orphaned files, missing masks).
        
        Returns:
            Validation report dictionary
        """
        if not self.cache_dir:
            return {'valid': True, 'issues': []}
        
        issues = []
        
        # Check for metadata files with missing masks
        for metadata_file in self.cache_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                task_id = metadata.get('task_id', metadata_file.stem.replace('_metadata', ''))
                material_counts = metadata.get('confidence_metrics', {}).get('material_counts', {})
                
                for material in material_counts.keys():
                    mask_path = self.cache_dir / f"{task_id}_{material}_mask.png"
                    if not mask_path.exists():
                        issues.append(f"Missing mask: {task_id}/{material}")
            except Exception as e:
                issues.append(f"Invalid metadata: {metadata_file.name} ({e})")
        
        # Check for orphaned mask files (no metadata)
        metadata_task_ids = set(
            f.stem.replace('_metadata', '')
            for f in self.cache_dir.glob("*_metadata.json")
        )
        
        for mask_file in self.cache_dir.glob("*_mask.png"):
            # Extract task_id from mask filename
            parts = mask_file.stem.rsplit('_', 2)
            if len(parts) >= 3:
                task_id = '_'.join(parts[:-2])
                if task_id not in metadata_task_ids:
                    issues.append(f"Orphaned mask: {mask_file.name}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues),
        }
