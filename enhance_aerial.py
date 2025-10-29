"""
Backward compatibility wrapper for enhance_aerial.

This module re-exports all functionality from board_material_aerial_enhancer
to maintain backward compatibility while avoiding code duplication.

All new code should import from board_material_aerial_enhancer directly.
"""

# Re-export all public APIs from board_material_aerial_enhancer
from board_material_aerial_enhancer import (
    DEFAULT_TEXTURES,
    MaterialRule,
    ClusterStats,
    compute_cluster_stats,
    relabel,
    relabel_safe,
    build_material_rules,
    save_palette_assignments,
    load_palette_assignments,
    auto_assign_materials_by_stats,
    enhance_aerial,
)

__all__ = [
    'DEFAULT_TEXTURES',
    'MaterialRule',
    'ClusterStats',
    'compute_cluster_stats',
    'relabel',
    'relabel_safe',
    'build_material_rules',
    'save_palette_assignments',
    'load_palette_assignments',
    'auto_assign_materials_by_stats',
    'enhance_aerial',
]
