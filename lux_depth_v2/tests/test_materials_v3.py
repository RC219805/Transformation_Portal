"""Unit tests for Materials V3 scaffolding.

Tests config validation, enum values, and basic structure.
Does NOT test implementation (NotImplementedError expected).
"""

import pytest

from lux_depth_v2.materials_v3 import (
    MaterialsV3Config,
    MaterialsV3Engine,
    MaterialTaxonomy,
    RefinementStrategy,
    ConfidenceSemantics,
    PromptGenerationConfig,
    EdgeAwareGating,
    ExpandedTaxonomyConfig,
)


def test_material_taxonomy_enum():
    """Test MaterialTaxonomy enum values."""
    assert MaterialTaxonomy.BASE == "base"
    assert MaterialTaxonomy.EXPANDED == "expanded"
    assert MaterialTaxonomy.FULL == "full"


def test_refinement_strategy_enum():
    """Test RefinementStrategy enum values."""
    assert RefinementStrategy.OFF == "off"
    assert RefinementStrategy.CANARY == "canary"
    assert RefinementStrategy.SELECTIVE == "selective"
    assert RefinementStrategy.AGGRESSIVE == "aggressive"


def test_confidence_semantics_defaults():
    """Test ConfidenceSemantics default values."""
    conf = ConfidenceSemantics()
    
    assert conf.base_threshold == 0.50
    assert conf.refined_threshold == 0.45
    assert conf.edge_threshold == 0.30
    assert conf.use_edge_confidence is True
    
    # Per-material thresholds
    assert conf.material_thresholds['glass'] == 0.40
    assert conf.material_thresholds['water'] == 0.35
    assert conf.material_thresholds['wood'] == 0.65


def test_confidence_semantics_get_threshold():
    """Test ConfidenceSemantics.get_threshold() logic."""
    conf = ConfidenceSemantics()
    
    # Base threshold for known material
    assert conf.get_threshold('glass', is_edge=False) == 0.40
    
    # Edge threshold (lowered)
    edge_thresh = conf.get_threshold('glass', is_edge=True)
    assert edge_thresh < 0.40
    assert edge_thresh >= conf.edge_threshold
    
    # Fallback to base_threshold for unknown material
    assert conf.get_threshold('unknown_material', is_edge=False) == conf.base_threshold


def test_prompt_generation_config_defaults():
    """Test PromptGenerationConfig default values."""
    cfg = PromptGenerationConfig()
    
    assert cfg.strategy == "mask_peaks"
    assert cfg.num_fg_points == 4
    assert cfg.fg_confidence_percentile == 80.0
    assert cfg.use_roi_crop is True
    assert cfg.fallback_to_box is True


def test_edge_aware_gating_defaults():
    """Test EdgeAwareGating default values."""
    gating = EdgeAwareGating()
    
    assert gating.enabled is True
    assert gating.core_threshold == 0.70
    assert gating.edge_low == 0.20
    assert gating.edge_high == 0.70
    assert gating.edge_method == "confidence_gradient"


def test_expanded_taxonomy_config_defaults():
    """Test ExpandedTaxonomyConfig default values."""
    tax = ExpandedTaxonomyConfig()
    
    assert tax.enabled is False
    assert 'sky' in tax.semantic_classes
    assert 'building' in tax.semantic_classes
    assert 'wood_grain' in tax.material_classes
    assert 'glass_clear' in tax.material_classes
    
    # Semantic→material mapping
    assert 'glass_clear' in tax.semantic_to_material_map.get('window', [])


def test_materials_v3_config_defaults():
    """Test MaterialsV3Config default values (disabled by default)."""
    cfg = MaterialsV3Config()
    
    assert cfg.enabled is False
    assert cfg.taxonomy == MaterialTaxonomy.BASE
    assert cfg.refine_edges == RefinementStrategy.OFF
    assert cfg.max_megapixels == 30.0
    assert cfg.max_dimension == 6000
    assert cfg.backend == 'segformer'
    assert cfg.lighting_aware is False


def test_materials_v3_config_with_enabled():
    """Test MaterialsV3Config when explicitly enabled."""
    cfg = MaterialsV3Config(
        enabled=True,
        taxonomy=MaterialTaxonomy.EXPANDED,
        refine_edges=RefinementStrategy.CANARY,
    )
    
    assert cfg.enabled is True
    assert cfg.taxonomy == MaterialTaxonomy.EXPANDED
    assert cfg.refine_edges == RefinementStrategy.CANARY


def test_materials_v3_engine_init_disabled():
    """Test MaterialsV3Engine initialization when disabled."""
    cfg = MaterialsV3Config(enabled=False)
    engine = MaterialsV3Engine(cfg)
    
    assert engine.config.enabled is False


def test_materials_v3_engine_init_enabled():
    """Test MaterialsV3Engine initialization when enabled."""
    cfg = MaterialsV3Config(enabled=True)
    engine = MaterialsV3Engine(cfg)
    
    assert engine.config.enabled is True


def test_materials_v3_engine_process_disabled_passthrough():
    """Test MaterialsV3Engine.process() when disabled (pass-through)."""
    import numpy as np
    
    cfg = MaterialsV3Config(enabled=False)
    engine = MaterialsV3Engine(cfg)
    
    # Mock inputs
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    seg_result = {"masks": {}, "metadata": {}}
    
    # Should pass through unchanged when disabled
    result = engine.process(image, seg_result)
    assert result is seg_result


def test_materials_v3_engine_process_enabled_not_implemented():
    """Test MaterialsV3Engine.process() when enabled (not yet implemented)."""
    import numpy as np
    
    cfg = MaterialsV3Config(enabled=True)
    engine = MaterialsV3Engine(cfg)
    
    # Mock inputs
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    seg_result = {"masks": {}, "metadata": {}}
    
    # Should raise NotImplementedError when enabled (scaffolding only)
    with pytest.raises(NotImplementedError, match="Materials V3 processing not yet implemented"):
        engine.process(image, seg_result)


def test_materials_v3_engine_get_v3_report():
    """Test MaterialsV3Engine.get_v3_report() structure."""
    cfg = MaterialsV3Config(
        enabled=True,
        taxonomy=MaterialTaxonomy.EXPANDED,
        refine_edges=RefinementStrategy.CANARY,
    )
    engine = MaterialsV3Engine(cfg)
    
    report = engine.get_v3_report()
    
    assert "enabled" in report
    assert "taxonomy" in report
    assert "refinement_strategy" in report
    assert "edge_gating_enabled" in report
    assert "lighting_aware" in report
    
    assert report["enabled"] is True
    assert report["taxonomy"] == "expanded"
    assert report["refinement_strategy"] == "canary"


def test_confidence_semantics_coverage_bounds():
    """Test coverage min/max sanity bounds."""
    conf = ConfidenceSemantics()
    
    assert 0.0 < conf.min_coverage < 1.0
    assert 0.0 < conf.max_coverage <= 1.0
    assert conf.min_coverage < conf.max_coverage
