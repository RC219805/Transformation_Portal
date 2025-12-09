"""Resource Monitor for Lux Depth V2 Pipeline.

Real-time monitoring of:
- MPS memory (Apple Silicon)
- CPU and RAM usage
- Disk space (internal + external storage)
- Performance metrics collection
"""

from __future__ import annotations

import platform
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import psutil

from .logging_utils import setup_logging


@dataclass
class ResourceThresholds:
    """Configurable thresholds for resource alerts."""
    mps_memory_gb: float = 55.0  # For 64GB unified memory, leave 9GB buffer
    cpu_percent: float = 90.0
    ram_percent: float = 85.0
    disk_space_gb: float = 10.0  # Minimum free space


@dataclass
class ResourceMetrics:
    """Snapshot of system resources."""
    timestamp: float
    
    # Memory
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    
    # CPU
    cpu_percent: float
    cpu_count: int
    
    # MPS (Apple Silicon)
    mps_available: bool = False
    mps_allocated_gb: Optional[float] = None
    mps_reserved_gb: Optional[float] = None
    
    # Disk
    disk_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Alerts
    alerts: List[str] = field(default_factory=list)


class ResourceMonitor:
    """Real-time resource monitoring with alerting.
    
    Features:
    - MPS memory tracking for Apple Silicon
    - CPU and RAM monitoring
    - Disk space tracking (internal + external)
    - Configurable alert thresholds
    - Performance metrics collection
    
    Args:
        alert_thresholds: Custom alert thresholds
        alert_callback: Optional callback for alerts
        logger: Optional logger instance
    """
    
    def __init__(
        self,
        alert_thresholds: Optional[ResourceThresholds] = None,
        alert_callback: Optional[Callable[[str], None]] = None,
        logger=None
    ):
        self.thresholds = alert_thresholds or ResourceThresholds()
        self.alert_callback = alert_callback
        self.logger = logger or setup_logging("INFO")
        
        # Detect system capabilities
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = self.is_mac and platform.processor() == "arm"
        
        # Try to import torch for MPS monitoring
        self.torch_available = False
        self.torch = None
        try:
            import torch
            self.torch = torch
            self.torch_available = True
            self.mps_available = (
                self.is_apple_silicon and 
                hasattr(torch.backends, "mps") and 
                torch.backends.mps.is_available()
            )
        except ImportError:
            self.mps_available = False
        
        # Metrics history
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history = 1000  # Keep last 1000 samples
        
        self.logger.info(
            f"ResourceMonitor initialized | "
            f"system={platform.system()} "
            f"arch={platform.machine()} "
            f"apple_silicon={self.is_apple_silicon} "
            f"mps_available={self.mps_available}"
        )
    
    def check_mps_memory(self) -> Dict[str, float]:
        """Check MPS memory usage (Apple Silicon only).
        
        Returns:
            Dictionary with allocated_gb and reserved_gb (or empty dict if N/A)
        """
        if not self.mps_available or not self.torch:
            return {}
        
        try:
            # Get MPS memory stats
            allocated = self.torch.mps.current_allocated_memory()
            reserved = self.torch.mps.driver_allocated_memory()
            
            return {
                "allocated_gb": allocated / (1024 ** 3),
                "reserved_gb": reserved / (1024 ** 3),
            }
        except Exception as e:
            self.logger.debug(f"MPS memory check failed: {e}")
            return {}
    
    def check_disk_space(self, paths: Optional[List[Path]] = None) -> Dict[str, Dict[str, float]]:
        """Check disk space for specified paths.
        
        Args:
            paths: List of paths to check (default: [cwd, /Volumes/T9 if exists])
            
        Returns:
            Dictionary mapping path -> {total_gb, used_gb, free_gb, percent}
        """
        if paths is None:
            paths = [Path.cwd()]
            # Check for T9 external drive
            t9_path = Path("/Volumes/T9")
            if t9_path.exists():
                paths.append(t9_path)
        
        result = {}
        for path in paths:
            try:
                usage = shutil.disk_usage(path)
                result[str(path)] = {
                    "total_gb": usage.total / (1024 ** 3),
                    "used_gb": usage.used / (1024 ** 3),
                    "free_gb": usage.free / (1024 ** 3),
                    "percent": (usage.used / usage.total) * 100,
                }
            except Exception as e:
                self.logger.debug(f"Disk check failed for {path}: {e}")
        
        return result
    
    def is_safe_to_process(
        self,
        image_size_mp: float,
        upscale: int = 4,
        strict: bool = False
    ) -> bool:
        """Pre-flight check: Is it safe to process this image?
        
        Args:
            image_size_mp: Input image size in megapixels
            upscale: Upscale factor (2 or 4)
            strict: If True, use stricter thresholds
            
        Returns:
            True if safe to process
        """
        metrics = self.get_metrics()
        
        # Estimate memory requirements
        # Rule of thumb: 4 bytes/pixel * upscale^2 * safety_factor
        output_mp = image_size_mp * (upscale ** 2)
        estimated_memory_gb = (output_mp * 4 * 3) / (1024 ** 3)  # RGB
        safety_factor = 1.5 if strict else 1.2
        required_memory_gb = estimated_memory_gb * safety_factor
        
        # Check MPS memory (if available)
        if metrics.mps_available and metrics.mps_allocated_gb is not None:
            mps_threshold = self.thresholds.mps_memory_gb
            if strict:
                mps_threshold *= 0.9  # More conservative
            
            if metrics.mps_allocated_gb + required_memory_gb > mps_threshold:
                self.logger.warning(
                    f"Insufficient MPS memory | "
                    f"allocated={metrics.mps_allocated_gb:.1f}GB "
                    f"required={required_memory_gb:.1f}GB "
                    f"threshold={mps_threshold:.1f}GB"
                )
                return False
        
        # Check RAM
        ram_threshold = self.thresholds.ram_percent
        if strict:
            ram_threshold *= 0.9
        
        if metrics.ram_percent > ram_threshold:
            self.logger.warning(
                f"High RAM usage | "
                f"current={metrics.ram_percent:.1f}% "
                f"threshold={ram_threshold:.1f}%"
            )
            return False
        
        # Check disk space
        for path, disk_metrics in metrics.disk_metrics.items():
            if disk_metrics["free_gb"] < self.thresholds.disk_space_gb:
                self.logger.warning(
                    f"Low disk space | "
                    f"path={path} "
                    f"free={disk_metrics['free_gb']:.1f}GB "
                    f"threshold={self.thresholds.disk_space_gb:.1f}GB"
                )
                return False
        
        return True
    
    def get_metrics(self) -> ResourceMetrics:
        """Get current resource metrics snapshot.
        
        Returns:
            ResourceMetrics object with current system state
        """
        # RAM
        mem = psutil.virtual_memory()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # MPS
        mps_metrics = self.check_mps_memory()
        
        # Disk
        disk_metrics = self.check_disk_space()
        
        # Create metrics object
        metrics = ResourceMetrics(
            timestamp=time.time(),
            ram_total_gb=mem.total / (1024 ** 3),
            ram_used_gb=mem.used / (1024 ** 3),
            ram_percent=mem.percent,
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            mps_available=self.mps_available,
            mps_allocated_gb=mps_metrics.get("allocated_gb"),
            mps_reserved_gb=mps_metrics.get("reserved_gb"),
            disk_metrics=disk_metrics,
        )
        
        # Check for alerts
        alerts = []
        
        if metrics.ram_percent > self.thresholds.ram_percent:
            alert = f"High RAM usage: {metrics.ram_percent:.1f}% (threshold: {self.thresholds.ram_percent}%)"
            alerts.append(alert)
        
        if metrics.cpu_percent > self.thresholds.cpu_percent:
            alert = f"High CPU usage: {metrics.cpu_percent:.1f}% (threshold: {self.thresholds.cpu_percent}%)"
            alerts.append(alert)
        
        if metrics.mps_available and metrics.mps_allocated_gb is not None:
            if metrics.mps_allocated_gb > self.thresholds.mps_memory_gb:
                alert = f"High MPS memory: {metrics.mps_allocated_gb:.1f}GB (threshold: {self.thresholds.mps_memory_gb}GB)"
                alerts.append(alert)
        
        for path, disk in disk_metrics.items():
            if disk["free_gb"] < self.thresholds.disk_space_gb:
                alert = f"Low disk space on {path}: {disk['free_gb']:.1f}GB (threshold: {self.thresholds.disk_space_gb}GB)"
                alerts.append(alert)
        
        metrics.alerts = alerts
        
        # Trigger alert callback
        if alerts and self.alert_callback:
            for alert in alerts:
                self.alert_callback(alert)
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        return metrics
    
    def get_summary(self) -> Dict[str, any]:
        """Get a human-readable summary of current resources.
        
        Returns:
            Dictionary with formatted resource information
        """
        metrics = self.get_metrics()
        
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metrics.timestamp)),
            "ram": {
                "used_gb": f"{metrics.ram_used_gb:.1f}",
                "total_gb": f"{metrics.ram_total_gb:.1f}",
                "percent": f"{metrics.ram_percent:.1f}%",
            },
            "cpu": {
                "percent": f"{metrics.cpu_percent:.1f}%",
                "count": metrics.cpu_count,
            },
        }
        
        if metrics.mps_available and metrics.mps_allocated_gb is not None:
            summary["mps"] = {
                "allocated_gb": f"{metrics.mps_allocated_gb:.1f}",
                "reserved_gb": f"{metrics.mps_reserved_gb:.1f}" if metrics.mps_reserved_gb else "N/A",
            }
        
        if metrics.disk_metrics:
            summary["disk"] = {
                path: f"{info['free_gb']:.1f}GB free ({info['percent']:.1f}% used)"
                for path, info in metrics.disk_metrics.items()
            }
        
        if metrics.alerts:
            summary["alerts"] = metrics.alerts
        
        return summary
    
    def log_metrics(self):
        """Log current metrics to logger."""
        summary = self.get_summary()
        
        msg_parts = [
            f"RAM: {summary['ram']['used_gb']}/{summary['ram']['total_gb']}GB ({summary['ram']['percent']})",
            f"CPU: {summary['cpu']['percent']}",
        ]
        
        if "mps" in summary:
            msg_parts.append(f"MPS: {summary['mps']['allocated_gb']}GB")
        
        if "disk" in summary:
            for path, info in summary["disk"].items():
                msg_parts.append(f"Disk({Path(path).name}): {info}")
        
        self.logger.info("Resources | " + " | ".join(msg_parts))
        
        # Log alerts as warnings
        if "alerts" in summary:
            for alert in summary["alerts"]:
                self.logger.warning(f"⚠️  {alert}")
