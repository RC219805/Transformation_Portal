"""Unit tests for Materials V3 taxonomy normalization."""

import pytest

from lux_depth_v2.materials_v3_taxonomy import (
    normalize_material_name,
    get_material_metadata,
    should_refine_material,
    normalize_material_dict,
    SEMANTIC_TO_CANONICAL,
    CANONICAL_MATERIALS,
    MaterialMetadata,
)


class TestNormalizeMaterialName:
    """Tests for material name normalization."""
    
    def test_water_variants(self):
        """Test water normalization."""
        assert normalize_material_name("water") == "water"
        assert normalize_material_name("pool_water") == "water"
        assert normalize_material_name("pool") == "water"
        assert normalize_material_name("ocean") == "water"
        assert normalize_material_name("sea") == "water"
        assert normalize_material_name("water_surface") == "water"
    
    def test_foliage_variants(self):
        """Test foliage normalization."""
        assert normalize_material_name("foliage") == "foliage"
        assert normalize_material_name("tree") == "foliage"
        assert normalize_material_name("trees") == "foliage"
        assert normalize_material_name("vegetation") == "foliage"
        assert normalize_material_name("grass") == "foliage"
        assert normalize_material_name("shrub") == "foliage"
        assert normalize_material_name("leaves") == "foliage"
    
    def test_glass_variants(self):
        """Test glass normalization."""
        assert normalize_material_name("glass") == "glass"
        assert normalize_material_name("window") == "glass"
        assert normalize_material_name("windows") == "glass"
        assert normalize_material_name("mirror") == "glass"
        assert normalize_material_name("glazing") == "glass"
    
    def test_wood_variants(self):
        """Test wood normalization."""
        assert normalize_material_name("wood") == "wood"
        assert normalize_material_name("wooden") == "wood"
        assert normalize_material_name("timber") == "wood"
        assert normalize_material_name("raw_wood") == "wood"
        assert normalize_material_name("laminate") == "wood"
    
    def test_metal_variants(self):
        """Test metal normalization."""
        assert normalize_material_name("metal") == "metal"
        assert normalize_material_name("steel") == "metal"
        assert normalize_material_name("stainless") == "metal"
        assert normalize_material_name("aluminum") == "metal"
        assert normalize_material_name("chrome") == "metal"
        assert normalize_material_name("brass") == "metal"
    
    def test_stone_variants(self):
        """Test stone normalization."""
        assert normalize_material_name("stone") == "stone"
        assert normalize_material_name("marble") == "stone"
        assert normalize_material_name("granite") == "stone"
        assert normalize_material_name("paver") == "stone"
        # Concrete/brick also map to stone for response purposes
        assert normalize_material_name("concrete") == "stone"
        assert normalize_material_name("brick") == "stone"
    
    def test_case_insensitive(self):
        """Test case-insensitive normalization."""
        assert normalize_material_name("WATER") == "water"
        assert normalize_material_name("Pool_Water") == "water"
        assert normalize_material_name("FOLIAGE") == "foliage"
    
    def test_unknown_material_passthrough(self):
        """Test unknown material names pass through."""
        result = normalize_material_name("exotic_unknown_material")
        assert result == "exotic_unknown_material"


class TestGetMaterialMetadata:
    """Tests for material metadata retrieval."""
    
    def test_glass_metadata(self):
        """Test glass has correct metadata."""
        meta = get_material_metadata("glass")
        assert meta.canonical_key == "glass"
        assert meta.confidence_threshold == 0.40
        assert meta.refinement_priority == 10
        assert meta.benefits_from_effsam is True
        assert meta.specular_sensitive is True
    
    def test_water_metadata(self):
        """Test water metadata."""
        meta = get_material_metadata("water")
        assert meta.canonical_key == "water"
        assert meta.confidence_threshold == 0.35
        assert meta.refinement_priority == 9
        assert meta.benefits_from_effsam is True
    
    def test_foliage_metadata(self):
        """Test foliage metadata."""
        meta = get_material_metadata("foliage")
        assert meta.canonical_key == "foliage"
        assert meta.refinement_priority == 8
        assert meta.benefits_from_effsam is True
    
    def test_wood_metadata(self):
        """Test wood metadata."""
        meta = get_material_metadata("wood")
        assert meta.canonical_key == "wood"
        assert meta.confidence_threshold == 0.65
        assert meta.refinement_priority == 4
        assert meta.benefits_from_effsam is False
    
    def test_sky_metadata(self):
        """Test sky metadata (never refine)."""
        meta = get_material_metadata("sky")
        assert meta.canonical_key == "sky"
        assert meta.refinement_priority == 0
        assert meta.benefits_from_effsam is False
    
    def test_semantic_normalization_in_metadata(self):
        """Test metadata lookup normalizes semantic names."""
        # These should all return same metadata
        meta_pool = get_material_metadata("pool_water")
        meta_ocean = get_material_metadata("ocean")
        meta_water = get_material_metadata("water")
        
        assert meta_pool.canonical_key == "water"
        assert meta_ocean.canonical_key == "water"
        assert meta_water.canonical_key == "water"
    
    def test_unknown_material_gets_defaults(self):
        """Test unknown materials get default metadata."""
        meta = get_material_metadata("super_exotic_material")
        assert meta.canonical_key == "super_exotic_material"
        assert meta.confidence_threshold == 0.50  # default
        assert meta.refinement_priority == 5  # default


class TestShouldRefineMaterial:
    """Tests for refinement decision logic."""
    
    def test_strategy_off(self):
        """Test 'off' strategy never refines."""
        assert should_refine_material("glass", refinement_strategy="off") is False
        assert should_refine_material("water", refinement_strategy="off") is False
        assert should_refine_material("wood", refinement_strategy="off") is False
    
    def test_strategy_canary(self):
        """Test 'canary' strategy (Stage 6 validated)."""
        # Should refine: glass, water, foliage
        assert should_refine_material("glass", refinement_strategy="canary") is True
        assert should_refine_material("water", refinement_strategy="canary") is True
        assert should_refine_material("foliage", refinement_strategy="canary") is True
        
        # Should NOT refine: wood, metal, stone
        assert should_refine_material("wood", refinement_strategy="canary") is False
        assert should_refine_material("metal", refinement_strategy="canary") is False
        assert should_refine_material("stone", refinement_strategy="canary") is False
    
    def test_strategy_canary_with_semantic_names(self):
        """Test canary strategy normalizes names."""
        # These should all be treated as refinable materials
        assert should_refine_material("pool_water", refinement_strategy="canary") is True
        assert should_refine_material("window", refinement_strategy="canary") is True
        assert should_refine_material("tree", refinement_strategy="canary") is True
    
    def test_strategy_selective(self):
        """Test 'selective' strategy (priority >= 6)."""
        # High priority: glass (10), water (9), foliage (8), metal (6)
        assert should_refine_material("glass", refinement_strategy="selective") is True
        assert should_refine_material("water", refinement_strategy="selective") is True
        assert should_refine_material("foliage", refinement_strategy="selective") is True
        assert should_refine_material("metal", refinement_strategy="selective") is True
        
        # Lower priority: wood (4), stone (3)
        assert should_refine_material("wood", refinement_strategy="selective") is False
        assert should_refine_material("stone", refinement_strategy="selective") is False
    
    def test_strategy_aggressive(self):
        """Test 'aggressive' strategy (refine most things)."""
        # Should refine most materials
        assert should_refine_material("glass", refinement_strategy="aggressive") is True
        assert should_refine_material("water", refinement_strategy="aggressive") is True
        assert should_refine_material("wood", refinement_strategy="aggressive") is True
        assert should_refine_material("metal", refinement_strategy="aggressive") is True
        
        # Should NOT refine: sky, ground, ceiling
        assert should_refine_material("sky", refinement_strategy="aggressive") is False
        assert should_refine_material("ground", refinement_strategy="aggressive") is False
        assert should_refine_material("ceiling", refinement_strategy="aggressive") is False
    
    def test_force_list_override(self):
        """Test explicit force_list overrides strategy."""
        force = {"wood", "stone"}
        
        # Wood and stone should refine despite 'canary' strategy
        assert should_refine_material("wood", refinement_strategy="canary", force_list=force) is True
        assert should_refine_material("stone", refinement_strategy="canary", force_list=force) is True
        
        # Glass should NOT refine if not in force list
        assert should_refine_material("glass", refinement_strategy="canary", force_list=force) is False


class TestNormalizeMaterialDict:
    """Tests for dictionary normalization."""
    
    def test_basic_normalization(self):
        """Test basic dictionary normalization."""
        d = {
            "pool_water": 0.8,
            "window": 0.6,
            "tree": 0.5,
        }
        normalized = normalize_material_dict(d)
        
        assert normalized == {
            "water": 0.8,
            "glass": 0.6,
            "foliage": 0.5,
        }
    
    def test_duplicate_canonical_keeps_max(self):
        """Test duplicate canonical keys keep max value."""
        d = {
            "pool_water": 0.6,
            "ocean": 0.8,  # both map to 'water'
            "water": 0.5,
        }
        normalized = normalize_material_dict(d)
        
        # Should keep max value (0.8)
        assert normalized["water"] == 0.8
    
    def test_mixed_semantic_canonical(self):
        """Test mixed semantic + canonical names."""
        d = {
            "glass": 0.7,
            "window": 0.5,  # also maps to glass → should keep 0.7
            "wood": 0.6,
            "timber": 0.8,  # also maps to wood → should keep 0.8
        }
        normalized = normalize_material_dict(d)
        
        assert normalized["glass"] == 0.7  # kept original
        assert normalized["wood"] == 0.8  # kept max
    
    def test_empty_dict(self):
        """Test empty dictionary."""
        assert normalize_material_dict({}) == {}


class TestTaxonomyConsistency:
    """Tests for taxonomy data consistency."""
    
    def test_all_mappings_point_to_canonical(self):
        """Test all semantic mappings point to known canonical keys."""
        for semantic, canonical in SEMANTIC_TO_CANONICAL.items():
            # Canonical key should exist (or be passthrough)
            # Most should be in CANONICAL_MATERIALS
            if canonical not in CANONICAL_MATERIALS:
                # It's OK if it's a deliberate alias
                assert canonical in {"wall", "floor", "ceiling", "ground", "sky", "stone"}
    
    def test_canonical_materials_are_strings(self):
        """Test all canonical materials are strings."""
        for mat in CANONICAL_MATERIALS:
            assert isinstance(mat, str)
            assert mat == mat.lower()  # Should be lowercase
            assert "_" in mat or mat.isalpha()  # snake_case or single word
