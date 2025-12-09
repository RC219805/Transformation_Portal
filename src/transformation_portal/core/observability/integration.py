"""
Observability integration helpers.

Provides simple integration with existing observability module.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Setup logging for a pipeline.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level
        log_file: Optional log file path
        json_format: Use JSON formatting (requires json_logging)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if json_format:
        try:
            # Try to use structured logging if available
            from lux_depth_v2.observability import json_logging
            formatter = json_logging.JSONFormatter()
        except ImportError:
            # Fallback to standard format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_metrics(enable: bool = True):
    """
    Setup metrics collection.
    
    Args:
        enable: Enable metrics collection
    """
    if not enable:
        return
    
    # Try to initialize observability metrics if available
    try:
        from lux_depth_v2.observability import metrics
        metrics.initialize()
    except ImportError:
        pass  # Observability not available


def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a simple logger (convenience function).
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Logger instance
    """
    return setup_logging(name, level)
