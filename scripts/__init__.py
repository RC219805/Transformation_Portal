"""Scripts package for Transformation Portal utilities.

This package contains utility scripts for codebase maintenance,
analysis, and auxiliary processing tasks.

Modules:
    - codebase_philosophy_auditor: Compatibility wrapper for canonical analyzer
    - decision_decay_dashboard: Compatibility wrapper for canonical dashboard
    - temporal_evolution: Compatibility wrapper for canonical roadmap helpers
    - download_depth_models: Download Depth Anything V2 models
    - download_samples: Download sample images for testing
    - install_models: Interactive model installation (Grade A+)
    - install_models_auto: Automated model installation with retry logic
    - create_board_textures: Generate MBAR material textures
    - run_aerial_enhancement: Batch aerial photo enhancement
    - verify_setup: Verify repository setup and dependencies

Execution:
    Scripts can be run directly or as modules:

    Direct execution:
        python scripts/decision_decay_dashboard.py

    Module execution (recommended for relative imports):
        python -m scripts.decision_decay_dashboard

Notes:
    - Scripts use lazy imports for faster startup (PIL, matplotlib, torch)
    - Most scripts support --help for usage information
    - Refer to each script's docstring for specific usage and examples

Version: 1.0.0
Author: Transformation Portal Team
"""

__version__ = "1.0.0"
__author__ = "Transformation Portal Team"

# Lazy imports to avoid loading heavy dependencies
__all__ = [
    "codebase_philosophy_auditor",
    "decision_decay_dashboard",
    "temporal_evolution",
    "download_depth_models",
    "download_samples",
    "install_models",
    "install_models_auto",
    "create_board_textures",
    "run_aerial_enhancement",
    "verify_setup",
]
