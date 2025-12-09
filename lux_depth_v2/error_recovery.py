"""Error Recovery System for Lux Depth V2 Pipeline.

Features:
- Exponential backoff retry strategy
- Fallback strategies (MPS→CPU, 4x→2x upscale, etc.)
- Error classification (transient vs permanent)
- Retry budget per task
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

from .logging_utils import setup_logging


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    TRANSIENT = "transient"  # Temporary, retry likely to succeed
    RESOURCE = "resource"    # Out of memory, disk space, etc.
    INPUT = "input"          # Invalid input file
    PERMANENT = "permanent"  # Unrecoverable error


@dataclass
class RetryStrategy:
    """Retry strategy configuration."""
    max_retries: int = 3
    backoff_base: float = 2.0
    max_delay_s: float = 300.0  # 5 minutes
    jitter: float = 0.1  # Add random jitter to avoid thundering herd


class ErrorRecovery:
    """Handles error recovery with retry logic and fallback strategies.
    
    Features:
    - Exponential backoff with jitter
    - Error classification
    - Fallback config generation
    - Retry budget enforcement
    
    Args:
        strategy: Retry strategy configuration
        logger: Optional logger instance
    """
    
    def __init__(
        self,
        strategy: Optional[RetryStrategy] = None,
        logger=None
    ):
        self.strategy = strategy or RetryStrategy()
        self.logger = logger or setup_logging("INFO")
        
        # Track retry attempts per task
        self.retry_counts: Dict[str, int] = {}
        
        self.logger.info(
            f"ErrorRecovery initialized | "
            f"max_retries={self.strategy.max_retries} "
            f"backoff_base={self.strategy.backoff_base}"
        )
    
    def classify_error(self, error: Exception) -> ErrorCategory:
        """Classify an error to determine if it's retryable.
        
        Args:
            error: Exception to classify
            
        Returns:
            ErrorCategory
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Resource errors (likely transient or need fallback)
        if any(keyword in error_str for keyword in [
            "out of memory", "oom", "cuda out of memory",
            "mps out of memory", "memory allocation",
            "disk space", "no space left"
        ]):
            return ErrorCategory.RESOURCE
        
        # Transient errors (network, temporary failures)
        if any(keyword in error_str for keyword in [
            "timeout", "connection", "temporary",
            "busy", "locked", "unavailable"
        ]):
            return ErrorCategory.TRANSIENT
        
        # Input errors (bad data, corrupted files)
        if any(keyword in error_str for keyword in [
            "cannot identify", "invalid image", "corrupt",
            "unsupported format", "decode error"
        ]):
            return ErrorCategory.INPUT
        
        # Network errors
        if error_type in ["ConnectionError", "TimeoutError", "URLError"]:
            return ErrorCategory.TRANSIENT
        
        # Default to permanent for unknown errors
        return ErrorCategory.PERMANENT
    
    def should_retry(
        self,
        error: Exception,
        task_id: str,
        attempt: int
    ) -> Tuple[bool, str]:
        """Determine if an error should be retried.
        
        Args:
            error: Exception that occurred
            task_id: Task identifier
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Tuple of (should_retry, reason)
        """
        # Check retry budget
        if attempt >= self.strategy.max_retries:
            return False, f"Max retries ({self.strategy.max_retries}) exceeded"
        
        # Classify error
        category = self.classify_error(error)
        
        # Never retry permanent or input errors
        if category == ErrorCategory.PERMANENT:
            return False, "Permanent error (not retryable)"
        
        if category == ErrorCategory.INPUT:
            return False, "Input error (bad data)"
        
        # Retry transient and resource errors
        if category in [ErrorCategory.TRANSIENT, ErrorCategory.RESOURCE]:
            return True, f"{category.value} error (retryable)"
        
        return False, "Unknown error category"
    
    def get_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter.
        
        Args:
            attempt: Attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        import random
        
        # Exponential backoff: base^attempt
        delay = self.strategy.backoff_base ** attempt
        
        # Cap at max delay
        delay = min(delay, self.strategy.max_delay_s)
        
        # Add jitter (±10%)
        jitter_range = delay * self.strategy.jitter
        jitter = random.uniform(-jitter_range, jitter_range)
        delay += jitter
        
        return max(0.1, delay)  # Minimum 100ms
    
    def get_fallback_config(
        self,
        original_config: Dict[str, Any],
        error: Exception,
        attempt: int
    ) -> Dict[str, Any]:
        """Generate fallback configuration for retry.
        
        Args:
            original_config: Original task configuration
            error: Exception that occurred
            attempt: Attempt number (0-indexed)
            
        Returns:
            Modified configuration for retry
        """
        config = original_config.copy()
        category = self.classify_error(error)
        
        # Resource errors: try reducing resource usage
        if category == ErrorCategory.RESOURCE:
            # Attempt 1: Reduce Materials v2 segmentation resolution
            if attempt == 0 and config.get("materials_v2_enabled"):
                if "max_segmentation_side" in config:
                    current_size = config.get("max_segmentation_side", 1536)
                    config["max_segmentation_side"] = max(512, current_size // 2)
                    self.logger.info(f"Fallback: Reducing Materials v2 segmentation to {config['max_segmentation_side']}px")
            
            # Attempt 2: Switch to CPU if using GPU
            elif attempt == 1 and config.get("device") in ["auto", "cuda", "mps"]:
                config["device"] = "cpu"
                self.logger.info("Fallback: Switching to CPU device")
            
            # Attempt 3: Disable Materials v2
            elif attempt == 2 and config.get("materials_v2_enabled"):
                config["materials_v2_enabled"] = False
                self.logger.info("Fallback: Disabling Materials v2")
            
            # Attempt 4: Reduce upscale factor
            elif attempt == 3 and config.get("upscale", 4) == 4:
                config["upscale"] = 2
                self.logger.info("Fallback: Reducing upscale 4x → 2x")
            
            # Attempt 5: Disable upscaling
            elif attempt >= 4:
                config["upscaler_backend"] = "none"
                config["upscale"] = 1
                self.logger.info("Fallback: Disabling upscaling")
        
        # Transient errors: just retry with same config
        elif category == ErrorCategory.TRANSIENT:
            self.logger.info("Fallback: Retrying with same configuration")
        
        return config
    
    def execute_with_retry(
        self,
        func: Callable,
        task_id: str,
        *args,
        **kwargs
    ) -> Tuple[Any, bool, Optional[str]]:
        """Execute a function with retry logic.
        
        Args:
            func: Function to execute
            task_id: Task identifier for tracking
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Tuple of (result, success, error_message)
        """
        last_error = None
        
        for attempt in range(self.strategy.max_retries + 1):
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Success
                if attempt > 0:
                    self.logger.info(
                        f"Retry successful | task_id={task_id} attempt={attempt}"
                    )
                
                return result, True, None
                
            except Exception as e:
                last_error = e
                
                # Check if should retry
                should_retry, reason = self.should_retry(e, task_id, attempt)
                
                if not should_retry:
                    self.logger.error(
                        f"Permanent failure | task_id={task_id} "
                        f"attempt={attempt} reason={reason} error={e}"
                    )
                    return None, False, str(e)
                
                # Calculate backoff delay
                delay = self.get_backoff_delay(attempt)
                
                self.logger.warning(
                    f"Retrying after error | task_id={task_id} "
                    f"attempt={attempt}/{self.strategy.max_retries} "
                    f"delay={delay:.1f}s error={e}"
                )
                
                # Wait before retry
                time.sleep(delay)
        
        # All retries exhausted
        error_msg = f"All retries exhausted: {last_error}"
        self.logger.error(f"Retry failed | task_id={task_id} error={error_msg}")
        
        return None, False, error_msg
    
    def reset_retry_count(self, task_id: str):
        """Reset retry count for a task.
        
        Args:
            task_id: Task identifier
        """
        if task_id in self.retry_counts:
            del self.retry_counts[task_id]
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get retry statistics.
        
        Returns:
            Dictionary with retry stats
        """
        total_tasks = len(self.retry_counts)
        total_retries = sum(self.retry_counts.values())
        
        return {
            "total_tasks_with_retries": total_tasks,
            "total_retry_attempts": total_retries,
            "average_retries": total_retries / max(1, total_tasks),
        }


def with_retry(
    error_recovery: ErrorRecovery,
    task_id: str
):
    """Decorator for automatic retry logic.
    
    Usage:
        @with_retry(error_recovery, "task_123")
        def process_image(path):
            ...
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            result, success, error = error_recovery.execute_with_retry(
                func, task_id, *args, **kwargs
            )
            if not success:
                raise RuntimeError(f"Task failed after retries: {error}")
            return result
        return wrapper
    return decorator


class MaterialsV2FallbackStrategy:
    """Fallback strategies specifically for Materials v2 failures.
    
    Features:
    - Progressive degradation (high→medium→low quality)
    - Backend fallback (ONNX→Heuristic)
    - Resolution reduction
    - Complete disable as last resort
    """
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logging("INFO")
    
    def get_fallback_config(
        self,
        error: Exception,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return fallback configuration based on error type.
        
        Args:
            error: Exception that occurred
            config: Current Materials v2 configuration
            
        Returns:
            Fallback configuration or None to disable Materials v2
        """
        error_str = str(error).lower()
        
        # Memory errors: reduce segmentation resolution
        if 'memory' in error_str or 'oom' in error_str:
            current_size = config.get('max_segmentation_side', 1536)
            new_size = current_size // 2
            
            if new_size >= 512:  # Minimum viable size
                fallback = config.copy()
                fallback['max_segmentation_side'] = new_size
                # Increase feathering to compensate for lower resolution
                fallback['edge_feather_radius'] = config.get('edge_feather_radius', 3) * 2
                
                self.logger.warning(
                    f"Materials v2 memory fallback: {current_size}px → {new_size}px"
                )
                return fallback
            else:
                self.logger.warning(
                    "Materials v2 resolution already at minimum; disabling"
                )
                return None
        
        # Segmentation errors: switch to heuristic backend
        elif 'segmentation' in error_str or 'model' in error_str:
            if config.get('backend') != 'heuristic':
                fallback = config.copy()
                fallback['backend'] = 'heuristic'
                
                self.logger.warning(
                    f"Materials v2 backend fallback: {config.get('backend')} → heuristic"
                )
                return fallback
            else:
                self.logger.warning(
                    "Materials v2 already using heuristic backend; disabling"
                )
                return None
        
        # Confidence errors: lower threshold
        elif 'confidence' in error_str or 'quality' in error_str:
            current_threshold = config.get('confidence_threshold', 0.6)
            new_threshold = max(0.3, current_threshold - 0.2)
            
            fallback = config.copy()
            fallback['confidence_threshold'] = new_threshold
            
            self.logger.warning(
                f"Materials v2 confidence fallback: {current_threshold} → {new_threshold}"
            )
            return fallback
        
        # Unknown error: disable Materials v2 as last resort
        else:
            self.logger.warning(
                f"Materials v2 unknown error ({error}); disabling as last resort"
            )
            return None
