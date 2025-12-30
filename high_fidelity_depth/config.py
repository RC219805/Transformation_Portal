#!/usr/bin/env python3
"""
Configuration for High Fidelity Depth Module
=============================================

Frozen configurations for scene classification and quality validation.
"""

from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    """Scene classifier V2 configuration (frozen after tuning)."""

    # Classifier version
    version: str = "v2"

    # Thresholds (tuned on 7-image validation set 2025-12-18)
    threshold_ratio_high: float = 10.0
    threshold_ratio_low: float = 5.0
    threshold_depth_var_low: float = 0.02
    threshold_depth_var_high: float = 0.03
    threshold_edge_density: float = 0.05

    # Metadata
    tuning_date: str = "2025-12-18"
    tuning_dataset: str = "7-image validation set"
    tuning_accuracy: float = 0.857  # 6/7 correct (expected)

    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "version": self.version,
            "thresholds": {
                "ratio_high": self.threshold_ratio_high,
                "ratio_low": self.threshold_ratio_low,
                "depth_var_low": self.threshold_depth_var_low,
                "depth_var_high": self.threshold_depth_var_high,
                "edge_density": self.threshold_edge_density,
            },
            "metadata": {
                "tuning_date": self.tuning_date,
                "tuning_dataset": self.tuning_dataset,
                "tuning_accuracy": self.tuning_accuracy,
            },
        }


# Default classifier configuration (frozen)
DEFAULT_CLASSIFIER_CONFIG = ClassifierConfig()
