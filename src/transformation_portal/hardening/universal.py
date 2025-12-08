"""
Universal hardening wrapper for any pipeline.

This module provides a generic hardening layer that can wrap any pipeline
to add security, reproducibility, and observability features.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol, runtime_checkable
from pathlib import Path
import time
import hashlib
import json
import logging
from dataclasses import dataclass, asdict

if TYPE_CHECKING:
    from lux_depth_v2.hardening.policy import HardeningPolicy


logger = logging.getLogger(__name__)


@runtime_checkable
class Pipeline(Protocol):
    """Protocol for any pipeline that can be hardened."""
    
    def process(self, input_path: Path, **kwargs) -> Any:
        """Process input and return result."""
        ...


@dataclass
class ProcessingReport:
    """Generic processing report for reproducibility."""
    
    run_id: str
    input_path: str
    config_hash: str
    duration_ms: Optional[float]
    success: bool
    error: Optional[str]
    meta: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class UniversalHardenedWrapper:
    """
    Universal hardening wrapper for any pipeline.
    
    This wrapper adds:
    - Input validation (file exists, extension, size)
    - Reproducibility reporting (config hash, runtime info)
    - Performance profiling (optional)
    - Error handling and logging
    
    Example:
        >>> from transformation_portal.hardening import UniversalHardenedWrapper
        >>> from lux_depth_v2.hardening.policy import HardeningPolicy
        >>> 
        >>> policy = HardeningPolicy.load()
        >>> wrapped = UniversalHardenedWrapper(my_pipeline, policy)
        >>> result = wrapped.process(input_path)
    """
    
    def __init__(
        self,
        pipeline: Pipeline,
        policy: Optional['HardeningPolicy'] = None,
        enable_profiling: bool = True,
        enable_stamping: bool = True,
        enable_input_validation: bool = True
    ):
        """
        Initialize universal hardening wrapper.
        
        Args:
            pipeline: Pipeline to wrap (must implement process method)
            policy: HardeningPolicy instance (optional, uses default if None)
            enable_profiling: Enable performance profiling
            enable_stamping: Enable reproducibility report stamping
            enable_input_validation: Enable input validation
        """
        self.pipeline = pipeline
        self.policy = policy
        self.enable_profiling = enable_profiling
        self.enable_stamping = enable_stamping
        self.enable_input_validation = enable_input_validation
        
        # Import policy if needed
        if self.policy is None and self.enable_input_validation:
            try:
                from lux_depth_v2.hardening.policy import HardeningPolicy
                self.policy = HardeningPolicy.load()
            except ImportError:
                logger.warning(
                    "HardeningPolicy not available, input validation disabled"
                )
                self.enable_input_validation = False
    
    def process(self, input_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Process with hardening applied.
        
        Args:
            input_path: Path to input file
            **kwargs: Additional arguments passed to wrapped pipeline
        
        Returns:
            Dictionary containing:
                - result: Output from wrapped pipeline
                - report: Reproducibility report (if stamping enabled)
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input validation fails
        """
        run_id = self._generate_run_id()
        input_path = Path(input_path)
        
        logger.info(
            "Processing with universal hardening",
            extra={"run_id": run_id, "input_path": str(input_path)}
        )
        
        # 1. Input validation
        if self.enable_input_validation:
            try:
                validated_path = self._validate_input(input_path)
            except Exception as e:
                return self._create_error_response(
                    run_id, input_path, kwargs, str(e)
                )
        else:
            validated_path = input_path
        
        # 2. Profiling start
        if self.enable_profiling:
            start = time.perf_counter()
        
        # 3. Execute wrapped pipeline
        try:
            result = self.pipeline.process(validated_path, **kwargs)
            success = True
            error = None
        except Exception as e:
            logger.error(
                f"Pipeline failed: {e}",
                extra={
                    "run_id": run_id,
                    "input_path": str(input_path),
                    "exception_type": type(e).__name__
                },
                exc_info=True
            )
            result = None
            success = False
            error = str(e)
        
        # 4. Profiling end
        if self.enable_profiling:
            duration_ms = (time.perf_counter() - start) * 1000
        else:
            duration_ms = None
        
        # 5. Report stamping
        if self.enable_stamping:
            report = self._create_report(
                run_id=run_id,
                input_path=input_path,
                result=result,
                duration_ms=duration_ms,
                success=success,
                error=error,
                kwargs=kwargs
            )
            return {"result": result, "report": report, "success": success}
        
        return {"result": result, "success": success}
    
    def _validate_input(self, input_path: Path) -> Path:
        """Validate input file."""
        if self.policy is None:
            return input_path
        
        # Use lux_depth_v2 hardening validation
        from lux_depth_v2.hardening.safe_io import validate_input_path
        
        try:
            return validate_input_path(input_path, self.policy)
        except Exception as e:
            raise ValueError(f"Input validation failed: {e}")
    
    def _create_report(
        self,
        run_id: str,
        input_path: Path,
        result: Any,
        duration_ms: Optional[float],
        success: bool,
        error: Optional[str],
        kwargs: Dict[str, Any]
    ) -> ProcessingReport:
        """Create reproducibility report."""
        config_hash = self._compute_config_hash(kwargs)
        
        # Gather metadata
        meta = self._gather_metadata()
        
        return ProcessingReport(
            run_id=run_id,
            input_path=str(input_path),
            config_hash=config_hash,
            duration_ms=duration_ms,
            success=success,
            error=error,
            meta=meta
        )
    
    def _create_error_response(
        self,
        run_id: str,
        input_path: Path,
        kwargs: Dict[str, Any],
        error: str
    ) -> Dict[str, Any]:
        """Create error response."""
        report = ProcessingReport(
            run_id=run_id,
            input_path=str(input_path),
            config_hash=self._compute_config_hash(kwargs),
            duration_ms=None,
            success=False,
            error=error,
            meta={}
        )
        return {"result": None, "report": report, "success": False}
    
    def _compute_config_hash(self, kwargs: Dict[str, Any]) -> str:
        """Compute deterministic config hash."""
        hasher = hashlib.sha256()
        config_str = json.dumps(kwargs, sort_keys=True, default=str)
        hasher.update(config_str.encode())
        return hasher.hexdigest()
    
    def _gather_metadata(self) -> Dict[str, Any]:
        """Gather runtime metadata."""
        try:
            from lux_depth_v2.hardening.runtime import gather_runtime_info
            runtime_info = gather_runtime_info()
        except ImportError:
            runtime_info = self._basic_runtime_info()
        
        try:
            from lux_depth_v2.hardening.stamping import get_git_info
            git_info = get_git_info()
        except ImportError:
            git_info = {}
        
        return {
            "runtime": runtime_info,
            "git": git_info,
            "wrapper_version": "2.0.0"
        }
    
    def _basic_runtime_info(self) -> Dict[str, Any]:
        """Basic runtime info without lux_depth_v2 dependencies."""
        import platform
        import sys
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine()
        }
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        import uuid
        return str(uuid.uuid4())


def wrap_function(
    func: Callable,
    policy: Optional['HardeningPolicy'] = None,
    **wrapper_kwargs
) -> UniversalHardenedWrapper:
    """
    Wrap a standalone function with hardening.
    
    Args:
        func: Function to wrap (must accept input_path as first arg)
        policy: HardeningPolicy instance
        **wrapper_kwargs: Additional arguments for UniversalHardenedWrapper
    
    Returns:
        UniversalHardenedWrapper instance
    
    Example:
        >>> def my_process_func(input_path, **kwargs):
        ...     return {"processed": True}
        >>> 
        >>> wrapped = wrap_function(my_process_func)
        >>> result = wrapped.process(input_path)
    """
    class FunctionAdapter:
        """Adapter to make function look like Pipeline."""
        
        def __init__(self, func):
            self.func = func
        
        def process(self, input_path: Path, **kwargs):
            return self.func(input_path, **kwargs)
    
    return UniversalHardenedWrapper(
        pipeline=FunctionAdapter(func),
        policy=policy,
        **wrapper_kwargs
    )
