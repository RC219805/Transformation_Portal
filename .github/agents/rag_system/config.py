"""
Configuration management for RAG System.

Loads configuration from YAML file with fallback to defaults.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from exceptions import ConfigurationError


# Default configuration (fallback if YAML not found)
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
        'bm25_k1': 1.5,
        'bm25_b': 0.75,
        'top_k_default': 10,
        'enable_vector_search': False,
        'vector_model': 'all-MiniLM-L6-v2',
        'query_cache_size': 100,
    },
    'reranker': {
        'exact_match_bonus': 2.0,
        'recency_bonus': 0.5,
        'code_quality_bonus': 0.3,
        'documentation_bonus': 0.2,
        'test_bonus': 0.1,
    },
    'citation': {
        'snippet_max_lines': 10,
        'snippet_max_chars': 500,
        'max_expected_score': 20.0,
        'default_max_citations': 5,
    },
    'logging': {
        'level': 'INFO',
        'log_to_file': False,
        'log_file': 'rag_system.log',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    },
    'knowledge_engine': {
        'min_samples_for_analysis': 5,
        'trend_window_days': 30,
        'confidence_threshold': 0.7,
    },
}


class Config:
    """
    Configuration manager for RAG system.

    Loads configuration from YAML file with environment variable overrides
    and fallback to defaults.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, looks for config.yaml
                        in the same directory as this module.
        """
        self._config: Dict[str, Any] = DEFAULT_CONFIG.copy()

        if config_path is None:
            # Default to config.yaml in same directory
            config_path = Path(__file__).parent / 'config.yaml'

        self.config_path = Path(config_path)
        self._load_config()
        self._apply_env_overrides()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logging.warning(
                f"Config file not found: {self.config_path}. Using defaults."
            )
            return

        try:
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)

            if user_config:
                self._merge_configs(self._config, user_config)

            logging.info(f"Loaded configuration from {self.config_path}")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load config from {self.config_path}: {e}"
            )

    def _merge_configs(self, base: Dict, override: Dict):
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Example: RAG_INDEXER_CACHE_ENABLED=false
        prefix = 'RAG_'

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Parse env key: RAG_INDEXER_CACHE_ENABLED -> ['indexer', 'cache_enabled']
            parts = env_key[len(prefix):].lower().split('_')

            if len(parts) >= 2:
                section = parts[0]
                key = '_'.join(parts[1:])

                if section in self._config and key in self._config[section]:
                    # Convert string to appropriate type
                    original_type = type(self._config[section][key])
                    try:
                        if original_type == bool:
                            self._config[section][key] = env_value.lower() in ('true', '1', 'yes')
                        elif original_type == int:
                            self._config[section][key] = int(env_value)
                        elif original_type == float:
                            self._config[section][key] = float(env_value)
                        else:
                            self._config[section][key] = env_value

                        logging.debug(f"Applied env override: {env_key}={env_value}")
                    except ValueError as e:
                        logging.warning(
                            f"Could not parse env var {env_key}={env_value}: {e}"
                        )

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            section: Configuration section (e.g., 'indexer')
            key: Configuration key (e.g., 'chunk_size_tokens')
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        return self._config.get(section, {}).get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name (e.g., 'indexer')

        Returns:
            Dictionary of configuration values
        """
        return self._config.get(section, {}).copy()

    def set(self, section: str, key: str, value: Any):
        """
        Set a configuration value (runtime only, not persisted).

        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance.

    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def reset_config():
    """Reset global configuration (useful for testing)."""
    global _global_config
    _global_config = None
