"""
Configuration management for RAG System.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .exceptions import ConfigError
from .logger import get_logger

logger = get_logger(__name__)


class Config:
    """
    Configuration manager for RAG system.

    Loads configuration from YAML file and provides type-safe access.
    """

    DEFAULT_CONFIG = {
        'indexer': {
            'chunk_size_tokens': 750,
            'overlap_tokens': 75,
            'chars_per_token': 4.0,
            'cache_enabled': True,
            'cache_dir': '.rag_cache',
        },
        'retriever': {
            'bm25_weight': 0.7,
            'vector_weight': 0.3,
            'enable_vector_search': False,
            'vector_model': 'all-MiniLM-L6-v2',
            'bm25_k1': 1.5,
            'bm25_b': 0.75,
            'query_cache_size': 100,
        },
        'citation': {
            'max_results': 5,
            'include_line_numbers': True,
            'max_expected_score': 20.0,
        },
        'reranker': {
            'enabled': False,
            'model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config: Dict[str, Any] = self.DEFAULT_CONFIG.copy()

        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
        else:
            # Try to find config.yaml in common locations
            search_paths = [
                'config.yaml',
                'rag_config.yaml',
                '.github/agents/rag_system/config.yaml',
            ]

            for path in search_paths:
                if Path(path).exists():
                    logger.info(f"Found config file: {path}")
                    self.load_from_file(path)
                    break
            else:
                logger.debug("No config file found, using defaults")

    def load_from_file(self, config_path: str):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)

            if file_config:
                # Merge with defaults (file config takes precedence)
                self._deep_merge(self.config, file_config)
                logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            raise ConfigError(f"Failed to load config: {e}")

    def _deep_merge(self, base: Dict, updates: Dict):
        """Deep merge updates into base dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, *args, **kwargs) -> Any:
        """
        Get a configuration value.

        Supports two calling conventions:
        - get('section.key', default=None) - dot notation
        - get('section', 'key', default=None) - separate args (legacy)

        Args:
            *args: Either (key_path, default) or (section, key, default)
            **kwargs: Optional 'default' keyword argument

        Returns:
            Configuration value
        """
        default = kwargs.get('default', None)

        # Handle both calling conventions
        if len(args) == 1:
            # Dot notation: config.get('indexer.chunk_size_tokens')
            key = args[0]
        elif len(args) == 2:
            # Could be: config.get('indexer.chunk_size_tokens', default)
            # Or: config.get('indexer', 'chunk_size_tokens')
            if isinstance(args[1], str):
                # Second arg is a string, so it's section, key
                key = f"{args[0]}.{args[1]}"
            else:
                # Second arg is default value
                key = args[0]
                default = args[1]
        elif len(args) == 3:
            # config.get('indexer', 'chunk_size_tokens', default)
            key = f"{args[0]}.{args[1]}"
            default = args[2]
        else:
            raise TypeError(f"get() takes 1-3 positional arguments but {len(args)} were given")

        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name (e.g., 'indexer', 'retriever')

        Returns:
            Configuration dictionary for the section
        """
        return self.config.get(section, {})

    def set(self, *args):
        """
        Set a configuration value.

        Supports two calling conventions:
        - set('section.key', value) - dot notation
        - set('section', 'key', value) - separate args (legacy)

        Args:
            *args: Either (key_path, value) or (section, key, value)
        """
        if len(args) == 2:
            # Dot notation: config.set('indexer.chunk_size_tokens', 1000)
            key, value = args
        elif len(args) == 3:
            # Legacy: config.set('indexer', 'chunk_size_tokens', 1000)
            key = f"{args[0]}.{args[1]}"
            value = args[2]
        else:
            raise TypeError(f"set() takes 2-3 positional arguments but {len(args)} were given")

        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get the full configuration as a dictionary."""
        return self.config.copy()


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Config instance
    """
    global _config

    if _config is None:
        _config = Config(config_path)

    return _config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
