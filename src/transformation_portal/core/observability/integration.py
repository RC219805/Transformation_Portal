"""
Observability Integration.

Provides unified configuration for logging and metrics collection.
Supports structured JSON logging for production and pretty printing for development.
"""

import logging
import sys
import json
from typing import Optional, Dict, Any, Union
from datetime import datetime

# Optional Rich integration for pretty console logs
try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class JsonFormatter(logging.Formatter):
    """Format logs as newline-delimited JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        
        # Merge extra context if available
        if hasattr(record, "context"):
            log_obj.update(record.context)
            
        # Handle exceptions
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Configure the root logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output structured JSON (good for Datadog/Splunk).
        log_file: Optional path to write logs to file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 1. Console Handler
    if json_format:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JsonFormatter())
    elif RICH_AVAILABLE:
        # "World-Class" Developer Experience
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True
        )
    else:
        # Fallback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
    root_logger.addHandler(console_handler)

    # 2. File Handler (Optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        # File logs are always JSON or detailed text for debugging
        file_handler.setFormatter(JsonFormatter() if json_format else logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root_logger.addHandler(file_handler)


def create_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)


class MetricsRegistry:
    """Simple singleton for tracking application metrics."""
    _metrics: Dict[str, Union[int, float]] = {}

    @classmethod
    def increment(cls, key: str, value: int = 1):
        cls._metrics[key] = cls._metrics.get(key, 0) + value

    @classmethod
    def gauge(cls, key: str, value: Union[int, float]):
        cls._metrics[key] = value

    @classmethod
    def get_all(cls) -> Dict[str, Union[int, float]]:
        return cls._metrics.copy()


def setup_metrics() -> None:
    """Initialize metrics collection."""
    # Placeholder for Prometheus/StatsD init
    pass
