"""
Scripts package for Transformation Portal.

This package contains utility scripts for:
- Board texture creation
- Evolutionary checkpoints
- Import migration
- Aerial enhancement
- Synthetic viewing
- Temporal evolution
- Material assignment visualization

Note: Individual scripts can be imported directly:
    from scripts import create_board_textures
    from scripts.synthetic_viewer import main
"""

# Explicit imports to make modules available via package
# Some imports may fail if optional dependencies are not installed
try:
    from . import create_board_textures
except ImportError:
    create_board_textures = None

try:
    from . import evolutionary_checkpoint
except ImportError:
    evolutionary_checkpoint = None

try:
    from . import migrate_imports
except ImportError:
    migrate_imports = None

try:
    from . import run_aerial_enhancement
except ImportError:
    run_aerial_enhancement = None

try:
    from . import synthetic_viewer
except ImportError:
    synthetic_viewer = None

try:
    from . import temporal_evolution
except ImportError:
    temporal_evolution = None

try:
    from . import visualize_material_assignments
except ImportError:
    visualize_material_assignments = None

__all__ = [
    'create_board_textures',
    'evolutionary_checkpoint',
    'migrate_imports',
    'run_aerial_enhancement',
    'synthetic_viewer',
    'temporal_evolution',
    'visualize_material_assignments',
]
