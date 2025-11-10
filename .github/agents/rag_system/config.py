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

        # Apply environment variable overrides
        self._apply_env_overrides()

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

    def _apply_env_overrides(self):
        """
        Apply environment variable overrides.
        
        Environment variables should be in the format:
        RAG_<SECTION>_<KEY>=<value>
        
        Examples:
            RAG_INDEXER_CACHE_ENABLED=false
            RAG_RETRIEVER_BM25_WEIGHT=0.8
            RAG_LOGGING_LEVEL=DEBUG
        """
        prefix = 'RAG_'
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
                
            # Remove prefix and split into parts
            key_parts = env_key[len(prefix):].lower().split('_')
            
            if len(key_parts) < 2:
                logger.warning(f"Invalid environment variable format: {env_key}")
                continue
            
            # First part is section, rest is the key
            section = key_parts[0]
            key_name = '_'.join(key_parts[1:])
            
            # Check if section exists in config
            if section not in self.config:
                logger.warning(f"Unknown config section in {env_key}: {section}")
                continue
            
            # Convert value to appropriate type
            try:
                converted_value = self._convert_env_value(env_value)
                
                # Set the value using dot notation
                config_key = f"{section}.{key_name}"
                self.set(config_key, converted_value)
                logger.info(f"Applied environment override: {env_key} = {converted_value}")
                
            except Exception as e:
                logger.warning(f"Failed to apply environment override {env_key}: {e}")

    def _convert_env_value(self, value: str) -> Any:
        """
        Convert environment variable string to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value (bool, int, float, or string)

        Notes:
            Only 'true', 'yes', 'on' (case-insensitive) are treated as boolean True.
            Only 'false', 'no', 'off' (case-insensitive) are treated as boolean False.
            '1' and '0' are parsed as integers if possible, not as booleans.
        """
        # Handle boolean values (do NOT treat '1'/'0' as bools)
        if value.lower() in ('true', 'yes', 'on'):
            return True
        elif value.lower() in ('false', 'no', 'off'):
            return False
        # Try to convert to number
        try:
            # Try int first
            if '.' not in value:
                return int(value)
            # Try float
            return float(value)
        except ValueError:
            # Return as string
            return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Dot-separated key path (e.g., 'indexer.chunk_size_tokens')
            default: Default value if key not found

        Returns:
            Configuration value
        """
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

    def set(self, key: str, value: Any):
        """
        Set a configuration value.

        Args:
            key: Dot-separated key path
            value: Value to set
        """
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
