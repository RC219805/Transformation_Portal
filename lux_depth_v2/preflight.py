"""Pre-flight Validation for Lux Depth V2 Pipeline.

Features:
- System requirements check (Python version, dependencies)
- Resource availability check (memory, disk, GPU)
- Input file validation (format, size, readability)
- Depth map availability check
- Configuration validation
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from .logging_utils import setup_logging


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    message: str
    severity: str = "error"  # 'error', 'warning', 'info'
    details: Dict[str, any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    passed: bool
    results: List[ValidationResult]
    timestamp: float
    
    def get_errors(self) -> List[ValidationResult]:
        """Get all error-level results."""
        return [r for r in self.results if r.severity == "error" and not r.passed]
    
    def get_warnings(self) -> List[ValidationResult]:
        """Get all warning-level results."""
        return [r for r in self.results if r.severity == "warning" and not r.passed]
    
    def summary(self) -> str:
        """Get a summary string."""
        errors = len(self.get_errors())
        warnings = len(self.get_warnings())
        
        if self.passed:
            return f"✅ Validation passed ({warnings} warnings)"
        else:
            return f"❌ Validation failed ({errors} errors, {warnings} warnings)"


class PreFlightValidator:
    """Pre-flight validation checks before processing.
    
    Features:
    - System requirements (Python, dependencies)
    - Resource availability (memory, disk, GPU)
    - Input validation (files, depth maps)
    - Configuration validation
    
    Args:
        logger: Optional logger instance
    """
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logging("INFO")
    
    def validate_system(self) -> ValidationResult:
        """Validate system requirements.
        
        Returns:
            ValidationResult
        """
        # Check Python version (requires 3.10+)
        py_version = sys.version_info
        required_version = (3, 10)
        
        if py_version < required_version:
            return ValidationResult(
                passed=False,
                message=f"Python {required_version[0]}.{required_version[1]}+ required",
                severity="error",
                details={
                    "current": f"{py_version.major}.{py_version.minor}.{py_version.micro}",
                    "required": f"{required_version[0]}.{required_version[1]}+",
                }
            )
        
        # Check for required dependencies
        missing_deps = []
        optional_deps = []
        
        required = ["numpy", "PIL", "torch"]
        optional = ["tifffile", "psutil"]
        
        for dep in required:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        for dep in optional:
            try:
                __import__(dep)
            except ImportError:
                optional_deps.append(dep)
        
        if missing_deps:
            return ValidationResult(
                passed=False,
                message=f"Missing required dependencies: {', '.join(missing_deps)}",
                severity="error",
                details={"missing": missing_deps}
            )
        
        details = {
            "python_version": f"{py_version.major}.{py_version.minor}.{py_version.micro}",
            "platform": platform.system(),
            "architecture": platform.machine(),
        }
        
        if optional_deps:
            details["missing_optional"] = optional_deps
        
        return ValidationResult(
            passed=True,
            message="System requirements satisfied",
            severity="info",
            details=details
        )
    
    def validate_resources(
        self,
        image_size_mp: float = 0.0,
        upscale: int = 4,
        device: str = "auto"
    ) -> ValidationResult:
        """Validate available resources.
        
        Args:
            image_size_mp: Expected image size in megapixels
            upscale: Upscale factor
            device: Target device
            
        Returns:
            ValidationResult
        """
        try:
            import psutil
        except ImportError:
            return ValidationResult(
                passed=True,
                message="Resource validation skipped (psutil not available)",
                severity="warning"
            )
        
        # Check RAM
        mem = psutil.virtual_memory()
        ram_available_gb = mem.available / (1024 ** 3)
        
        # Estimate required memory (rough approximation)
        # 4 bytes/pixel * 3 channels * upscale^2 * safety_factor
        if image_size_mp > 0:
            estimated_memory_gb = (image_size_mp * 4 * 3 * (upscale ** 2)) / (1024 ** 3)
            estimated_memory_gb *= 1.5  # Safety factor
        else:
            estimated_memory_gb = 4.0  # Default estimate for unknown size
        
        if ram_available_gb < estimated_memory_gb:
            return ValidationResult(
                passed=False,
                message=f"Insufficient RAM: {ram_available_gb:.1f}GB available, {estimated_memory_gb:.1f}GB required",
                severity="error",
                details={
                    "available_gb": ram_available_gb,
                    "required_gb": estimated_memory_gb,
                }
            )
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 ** 3)
        min_disk_gb = 10.0
        
        if disk_free_gb < min_disk_gb:
            return ValidationResult(
                passed=False,
                message=f"Low disk space: {disk_free_gb:.1f}GB available, {min_disk_gb:.1f}GB recommended",
                severity="warning",
                details={
                    "free_gb": disk_free_gb,
                    "recommended_gb": min_disk_gb,
                }
            )
        
        # Check GPU availability
        gpu_info = self._check_gpu(device)
        
        return ValidationResult(
            passed=True,
            message="Resources available",
            severity="info",
            details={
                "ram_available_gb": ram_available_gb,
                "disk_free_gb": disk_free_gb,
                "gpu": gpu_info,
            }
        )
    
    def _check_gpu(self, device: str) -> Dict[str, any]:
        """Check GPU availability and capabilities."""
        try:
            import torch
            
            info = {
                "requested": device,
                "available": False,
                "type": None,
            }
            
            if device in ["auto", "cuda"]:
                if torch.cuda.is_available():
                    info["available"] = True
                    info["type"] = "cuda"
                    info["count"] = torch.cuda.device_count()
                    info["device_name"] = torch.cuda.get_device_name(0)
            
            if device in ["auto", "mps"]:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    info["available"] = True
                    info["type"] = "mps"
            
            if device == "cpu":
                info["available"] = True
                info["type"] = "cpu"
                if PSUTIL_AVAILABLE and psutil:
                    info["count"] = psutil.cpu_count()
            
            return info
            
        except ImportError:
            return {"error": "torch not available"}
    
    def validate_input_file(
        self,
        input_path: Path,
        max_size_mp: float = 324.0  # 324MP limit (18K x 18K)
    ) -> ValidationResult:
        """Validate a single input file.
        
        Args:
            input_path: Path to input file
            max_size_mp: Maximum image size in megapixels
            
        Returns:
            ValidationResult
        """
        # Check file exists
        if not input_path.exists():
            return ValidationResult(
                passed=False,
                message=f"Input file not found: {input_path}",
                severity="error"
            )
        
        # Check file is readable
        if not input_path.is_file():
            return ValidationResult(
                passed=False,
                message=f"Input path is not a file: {input_path}",
                severity="error"
            )
        
        # Check file extension
        valid_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"}
        if input_path.suffix.lower() not in valid_extensions:
            return ValidationResult(
                passed=False,
                message=f"Unsupported file format: {input_path.suffix}",
                severity="error",
                details={"valid_formats": list(valid_extensions)}
            )
        
        # Try to open and validate image
        try:
            from PIL import Image
            
            with Image.open(input_path) as img:
                width, height = img.size
                megapixels = (width * height) / (1024 ** 2)
                
                if megapixels > max_size_mp:
                    return ValidationResult(
                        passed=False,
                        message=f"Image too large: {megapixels:.1f}MP (max {max_size_mp}MP)",
                        severity="error",
                        details={
                            "width": width,
                            "height": height,
                            "megapixels": megapixels,
                            "max_megapixels": max_size_mp,
                        }
                    )
                
                return ValidationResult(
                    passed=True,
                    message=f"Valid input file: {width}x{height} ({megapixels:.1f}MP)",
                    severity="info",
                    details={
                        "width": width,
                        "height": height,
                        "megapixels": megapixels,
                        "format": img.format,
                        "mode": img.mode,
                    }
                )
                
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Failed to read image: {e}",
                severity="error"
            )
    
    def validate_depth_map(
        self,
        input_path: Path,
        depth_dir: Optional[Path]
    ) -> ValidationResult:
        """Validate depth map availability.
        
        Args:
            input_path: Input image path
            depth_dir: Directory containing depth maps
            
        Returns:
            ValidationResult
        """
        if not depth_dir:
            return ValidationResult(
                passed=True,
                message="No depth directory specified (will process without depth)",
                severity="info"
            )
        
        if not depth_dir.exists():
            return ValidationResult(
                passed=False,
                message=f"Depth directory not found: {depth_dir}",
                severity="warning"
            )
        
        # Look for corresponding depth map
        stem = input_path.stem
        depth_found = False
        depth_path = None
        
        # Try both {stem}_depth.{ext} and {stem}.{ext} patterns (matches _find_depth in pipeline.py)
        for pattern in (f"{stem}_depth", f"{stem}"):
            for ext in [".tif", ".tiff", ".png"]:
                candidate = depth_dir / f"{pattern}{ext}"
                if candidate.exists():
                    depth_found = True
                    depth_path = candidate
                    break
            if depth_found:
                break
        
        if not depth_found:
            return ValidationResult(
                passed=False,
                message=f"Depth map not found for {input_path.name}",
                severity="warning",
                details={"searched_stem": stem, "depth_dir": str(depth_dir)}
            )
        
        # Validate depth map is readable
        try:
            from PIL import Image
            with Image.open(depth_path) as img:
                width, height = img.size
                
                return ValidationResult(
                    passed=True,
                    message=f"Valid depth map: {width}x{height}",
                    severity="info",
                    details={
                        "depth_path": str(depth_path),
                        "width": width,
                        "height": height,
                    }
                )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Failed to read depth map: {e}",
                severity="error"
            )
    
    def validate_materials_v2_config(
        self,
        config: Optional[Dict[str, any]]
    ) -> ValidationResult:
        """Validate Materials v2 configuration.
        
        Args:
            config: Materials v2 configuration dictionary
            
        Returns:
            ValidationResult
        """
        if not config or not config.get('enabled'):
            return ValidationResult(
                passed=True,
                message="Materials v2 disabled (skipping validation)",
                severity="info"
            )
        
        issues = []
        
        # Check backend availability
        backend = config.get('backend', 'heuristic')
        if backend == 'onnx':
            try:
                import onnxruntime
            except ImportError:
                issues.append("ONNX backend requested but onnxruntime not installed")
        
        # Check cache directory writable
        if config.get('cache_enabled'):
            cache_dir = config.get('cache_dir')
            if cache_dir:
                cache_path = Path(cache_dir)
                try:
                    cache_path.mkdir(parents=True, exist_ok=True)
                    # Test write
                    test_file = cache_path / ".test_write"
                    test_file.touch()
                    test_file.unlink()
                except Exception as e:
                    issues.append(f"Cache directory not writable: {cache_dir} ({e})")
        
        # Check confidence threshold valid
        confidence_config = config.get('confidence', {})
        threshold = confidence_config.get('confidence_threshold', 0.6)
        if not 0.0 <= threshold <= 1.0:
            issues.append(f"Invalid confidence threshold: {threshold} (must be 0.0-1.0)")
        
        # Check segmentation size valid
        seg_config = config.get('segmentation', {})
        max_seg_side = seg_config.get('max_segmentation_side', 1536)
        min_seg_side = seg_config.get('min_segmentation_side', 512)
        
        if max_seg_side < min_seg_side:
            issues.append(
                f"Invalid segmentation sizes: max={max_seg_side} < min={min_seg_side}"
            )
        
        if issues:
            return ValidationResult(
                passed=False,
                message=f"Materials v2 configuration issues: {'; '.join(issues)}",
                severity="error",
                details={"issues": issues}
            )
        
        return ValidationResult(
            passed=True,
            message="Materials v2 configuration valid",
            severity="info",
            details={
                "backend": backend,
                "confidence_threshold": threshold,
                "cache_enabled": config.get('cache_enabled', False),
                "max_segmentation_side": max_seg_side,
            }
        )
    
    def validate_all(
        self,
        input_path: Path,
        depth_dir: Optional[Path] = None,
        device: str = "auto",
        upscale: int = 4,
        materials_v2_config: Optional[Dict[str, any]] = None
    ) -> ValidationReport:
        """Comprehensive validation of all aspects.
        
        Args:
            input_path: Input image path
            depth_dir: Optional depth directory
            device: Target device
            upscale: Upscale factor
            
        Returns:
            ValidationReport
        """
        import time
        
        results = []
        
        # System validation
        results.append(self.validate_system())
        
        # Input file validation
        input_result = self.validate_input_file(input_path)
        results.append(input_result)
        
        # Extract image size for resource validation
        image_size_mp = 0.0
        if input_result.passed and "megapixels" in input_result.details:
            image_size_mp = input_result.details["megapixels"]
        
        # Resource validation
        results.append(self.validate_resources(image_size_mp, upscale, device))
        
        # Depth map validation (warning only, not fatal)
        results.append(self.validate_depth_map(input_path, depth_dir))
        
        # Materials v2 validation (if enabled)
        if materials_v2_config:
            results.append(self.validate_materials_v2_config(materials_v2_config))
        
        # Overall pass/fail (only errors count)
        errors = [r for r in results if r.severity == "error" and not r.passed]
        passed = len(errors) == 0
        
        report = ValidationReport(
            passed=passed,
            results=results,
            timestamp=time.time()
        )
        
        return report
    
    def log_report(self, report: ValidationReport):
        """Log validation report to logger.
        
        Args:
            report: ValidationReport to log
        """
        self.logger.info(f"Pre-flight validation | {report.summary()}")
        
        for result in report.results:
            level = "ERROR" if result.severity == "error" else "WARNING" if result.severity == "warning" else "INFO"
            status = "✅" if result.passed else "❌"
            
            if not result.passed or result.severity != "info":
                self.logger.info(f"  {status} [{level}] {result.message}")
