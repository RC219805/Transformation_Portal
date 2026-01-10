"""
Tests for run card generation and scene type taxonomy.
"""

import pytest
import yaml
from pathlib import Path
import tempfile
import shutil

# Import modules under test
from src.transformation_portal.scene_types import (
    normalize_scene_type,
    validate_scene_type,
    get_scene_type_description,
    list_scene_types,
    get_all_aliases
)


class TestSceneTypes:
    """Test scene type taxonomy module."""

    def test_normalize_exact_match(self):
        """Test normalization with exact canonical name."""
        assert normalize_scene_type("interior_kitchen") == "interior_kitchen"
        assert normalize_scene_type("exterior_pool") == "exterior_pool"

    def test_normalize_from_alias(self):
        """Test normalization from common aliases."""
        assert normalize_scene_type("kitchen") == "interior_kitchen"
        assert normalize_scene_type("pool") == "exterior_pool"
        assert normalize_scene_type("master") == "interior_bedroom"
        assert normalize_scene_type("drone") == "aerial_exterior"

    def test_normalize_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert normalize_scene_type("KITCHEN") == "interior_kitchen"
        assert normalize_scene_type("Pool") == "exterior_pool"
        assert normalize_scene_type("MaStEr") == "interior_bedroom"

    def test_normalize_with_whitespace(self):
        """Test normalization handles whitespace."""
        assert normalize_scene_type("  kitchen  ") == "interior_kitchen"
        assert normalize_scene_type("\tpool\n") == "exterior_pool"

    def test_normalize_invalid_type(self):
        """Test normalization raises ValueError for unknown types."""
        with pytest.raises(ValueError, match="Unknown scene type"):
            normalize_scene_type("invalid_type")

    def test_validate_scene_type_valid(self):
        """Test validation accepts valid canonical types."""
        assert validate_scene_type("interior_kitchen") is True
        assert validate_scene_type("exterior_pool") is True
        assert validate_scene_type("aerial_exterior") is True

    def test_validate_scene_type_invalid(self):
        """Test validation rejects invalid types."""
        assert validate_scene_type("invalid_type") is False
        assert validate_scene_type("kitchen") is False  # Alias, not canonical

    def test_get_scene_type_description(self):
        """Test description retrieval."""
        desc = get_scene_type_description("interior_kitchen")
        assert "Kitchen" in desc or "kitchen" in desc.lower()

        desc = get_scene_type_description("exterior_pool")
        assert "pool" in desc.lower() or "water" in desc.lower()

    def test_get_scene_type_description_invalid(self):
        """Test description retrieval for invalid type."""
        assert get_scene_type_description("invalid_type") is None

    def test_list_scene_types(self):
        """Test scene type listing."""
        types = list_scene_types()
        assert "interior_kitchen" in types
        assert "exterior_pool" in types
        assert "aerial_exterior" in types
        assert len(types) >= 15  # Should have at least 15 scene types

    def test_get_all_aliases(self):
        """Test alias retrieval."""
        aliases = get_all_aliases("interior_kitchen")
        assert "kitchen" in aliases
        assert "kit" in aliases

        aliases = get_all_aliases("exterior_pool")
        assert "pool" in aliases
        assert "water" in aliases


class TestRunCardGeneration:
    """Test run card generation functionality."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_infer_scene_type_kitchen(self):
        """Test scene type inference for kitchen images."""
        from scripts.utilities.generate_run_card import infer_scene_type

        assert infer_scene_type("renders/kitchen.jpg") == "interior_kitchen"
        assert infer_scene_type("750picacho/Kit_01.jpg") == "interior_kitchen"
        assert infer_scene_type("project/butler_pantry.jpg") == "interior_kitchen"

    def test_infer_scene_type_pool(self):
        """Test scene type inference for pool images."""
        from scripts.utilities.generate_run_card import infer_scene_type

        assert infer_scene_type("exterior/pool_view.jpg") == "exterior_pool"
        assert infer_scene_type("renders/Pool.jpg") == "exterior_pool"
        assert infer_scene_type("spa_area.jpg") == "exterior_pool"

    def test_infer_scene_type_aerial(self):
        """Test scene type inference for aerial images."""
        from scripts.utilities.generate_run_card import infer_scene_type

        assert infer_scene_type("drone/aerial_01.jpg") == "aerial_exterior"
        assert infer_scene_type("overhead_view.jpg") == "aerial_exterior"

    def test_infer_scene_type_bedroom(self):
        """Test scene type inference for bedroom images."""
        from scripts.utilities.generate_run_card import infer_scene_type

        assert infer_scene_type("master_bedroom.jpg") == "interior_bedroom"
        assert infer_scene_type("guest_suite.jpg") == "interior_bedroom"

    def test_generate_run_card_basic(self, temp_output_dir):
        """Test basic run card generation."""
        from scripts.utilities.generate_run_card import generate_run_card

        output_path = generate_run_card(
            image_path="test_images/kitchen.jpg",
            baseline_score=58.3,
            processed_score=54.1,
            recipe_name="signature_estate",
            project_name="test_project",
            output_dir=temp_output_dir
        )

        # Check file was created
        assert output_path.exists()
        assert output_path.name == "kitchen_signature_estate.yaml"

        # Load and validate content
        with open(output_path) as f:
            run_card = yaml.safe_load(f)

        assert run_card["image_id"] == "kitchen"
        assert run_card["baseline_score"] == 58.3
        assert run_card["processed_score"] == 54.1
        assert run_card["delta_score"] == -4.2
        assert run_card["recipe"] == "signature_estate"
        assert run_card["project"] == "test_project"
        assert run_card["scene_type"] == "interior_kitchen"

    def test_generate_run_card_with_settings(self, temp_output_dir):
        """Test run card generation with recipe settings."""
        from scripts.utilities.generate_run_card import generate_run_card

        recipe_settings = {"clarity": 0.2, "glow": 0.1, "saturation": 1.05}

        output_path = generate_run_card(
            image_path="test_images/pool.jpg",
            baseline_score=55.0,
            processed_score=58.0,
            recipe_name="pool_estate",
            project_name="villa_project",
            recipe_settings=recipe_settings,
            processing_time=45.3,
            output_dir=temp_output_dir
        )

        # Load and validate
        with open(output_path) as f:
            run_card = yaml.safe_load(f)

        assert run_card["recipe_settings"] == recipe_settings
        assert run_card["processing_time_seconds"] == 45.3
        assert run_card["scene_type"] == "exterior_pool"

    def test_generate_run_card_with_override(self, temp_output_dir):
        """Test run card generation with scene type override."""
        from scripts.utilities.generate_run_card import generate_run_card

        output_path = generate_run_card(
            image_path="test_images/unknown.jpg",
            baseline_score=60.0,
            processed_score=62.0,
            recipe_name="test_recipe",
            project_name="test_project",
            scene_type_override="interior_great_room",
            output_dir=temp_output_dir
        )

        with open(output_path) as f:
            run_card = yaml.safe_load(f)

        assert run_card["scene_type"] == "interior_great_room"

    def test_generate_run_card_project_subdirectory(self, temp_output_dir):
        """Test that run cards are organized by project."""
        from scripts.utilities.generate_run_card import generate_run_card

        output_path = generate_run_card(
            image_path="kitchen.jpg",
            baseline_score=58.0,
            processed_score=60.0,
            recipe_name="test",
            project_name="750_picacho",
            output_dir=temp_output_dir
        )

        # Check project subdirectory was created
        assert "750_picacho" in str(output_path)
        assert output_path.parent.name == "750_picacho"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
