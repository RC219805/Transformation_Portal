"""
Logging configuration for RAG System.

Provides centralized logging with file and console outputs.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config import get_config


# Logging level mapping
LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


def setup_logging(
    name: str = 'rag_system',
    config_override: Optional[dict] = None,
) -> logging.Logger:
    """
    Setup logging for RAG system.

    Args:
        name: Logger name
        config_override: Optional config dict to override defaults

    Returns:
        Configured logger instance
    """
    # Get configuration
    config = get_config()
    log_config = config.get_section('logging')

    if config_override:
        log_config.update(config_override)

    # Create logger
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set level
    level_str = log_config.get('level', 'INFO')
    level = LEVEL_MAP.get(level_str.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if enabled)
    if log_config.get('log_to_file', False):
        cache_dir = config.get('indexer', 'cache_dir', '.rag_cache')
        log_file = Path(cache_dir) / log_config.get('log_file', 'rag_system.log')

        # Create cache directory if needed
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


def get_logger(name: str = 'rag_system') -> logging.Logger:
    """
    Get or create logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # Setup if not already configured
    if not logger.handlers:
        return setup_logging(name)

    return logger
