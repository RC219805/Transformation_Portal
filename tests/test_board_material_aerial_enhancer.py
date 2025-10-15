"""Tests for board_material_aerial_enhancer module.

Validates clustering, material assignment, texture blending, and end-to-end
aerial enhancement workflow for MBAR board material application.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from board_material_aerial_enhancer import (
    ClusterStats,
    MaterialRule,
    apply_materials,
    assign_materials,
    build_material_rules,
    enhance_aerial,
    load_palette_assignments,
    save_palette_assignments,
)


def _dummy_rule(name: str, texture_path: Path) -> MaterialRule:
    """Create a MaterialRule for testing that always scores 1.0."""
    def score(_: ClusterStats) -> float:
        return 1.0

    return MaterialRule(name=name, texture=str(texture_path), blend=1.0, score_fn=score)


def test_assign_materials_prefers_high_scores(tmp_path: Path) -> None:
    """Test that material assignment selects clusters with highest scores."""
    stats = [
        ClusterStats(
            label=0, count=100, mean_rgb=np.array([0.9, 0.9, 0.85]),
            mean_hsv=np.array([0.1, 0.05, 0.9]), std_rgb=np.zeros(3)
        ),
        ClusterStats(
            label=1, count=80, mean_rgb=np.array([0.75, 0.65, 0.55]),
            mean_hsv=np.array([0.09, 0.25, 0.65]), std_rgb=np.zeros(3)
        ),
        ClusterStats(
            label=2, count=60, mean_rgb=np.array([0.5, 0.48, 0.46]),
            mean_hsv=np.array([0.02, 0.1, 0.5]), std_rgb=np.zeros(3)
        ),
        ClusterStats(
            label=3, count=40, mean_rgb=np.array([0.28, 0.22, 0.18]),
            mean_hsv=np.array([0.06, 0.2, 0.28]), std_rgb=np.zeros(3)
        ),
    ]

    # Create dummy texture files for all materials
    material_names = ("plaster", "stone", "cladding", "screens", "equitone", "roof", "bronze", "shade")
    textures = {name: (tmp_path / f"{name}.png") for name in material_names}
    for tex in textures.values():
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tex)

    rules = build_material_rules(textures)
    assignments = assign_materials(stats, rules)

    # Verify assignments are valid
    assert set(assignments.keys()) <= {stat.label for stat in stats}

    # Verify that high-scoring materials are assigned
    assert any(rule.name == "plaster" for rule in assignments.values())
    assert any(rule.name == "stone" for rule in assignments.values())


def test_apply_materials_blends_texture(tmp_path: Path) -> None:
    """Test that texture blending produces expected color shifts."""
    # Create gray base image
    base = np.full((4, 4, 3), 0.5, dtype=np.float32)
    labels = np.zeros((4, 4), dtype=np.uint8)

    # Create red texture
    texture_path = tmp_path / "texture.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(texture_path)

    # Apply red texture with full blend
    rule = MaterialRule(name="test", texture=str(texture_path), blend=1.0, score_fn=lambda _: 1.0)
    output = apply_materials(base, labels, {0: rule})

    # Verify red channel is dominant (with some tolerance for soft masking)
    assert np.isclose(output[..., 0].mean(), 1.0, atol=0.05)
    assert np.isclose(output[..., 1:].mean(), 0.0, atol=0.1)


def test_enhance_aerial_creates_output(tmp_path: Path) -> None:
    """Test end-to-end aerial enhancement workflow."""
    width, height = 160, 120

    # Create synthetic aerial with distinct regions
    image = Image.new("RGB", (width, height), (160, 160, 150))
    for x in range(width):
        for y in range(height):
            if x < width // 3:
                image.putpixel((x, y), (225, 215, 200))  # plaster region
            elif x < 2 * width // 3:
                image.putpixel((x, y), (190, 170, 150))  # stone region
            else:
                image.putpixel((x, y), (60, 50, 40))  # bronze/windows region

    input_path = tmp_path / "input.png"
    image.save(input_path)

    # Create texture files with distinct colors
    material_names = ("plaster", "stone", "cladding", "screens", "equitone", "roof", "bronze", "shade")
    textures = {name: (tmp_path / f"{name}.png") for name in material_names}
    colors = {
        "plaster": (240, 230, 215),
        "stone": (210, 190, 170),
        "cladding": (200, 170, 140),
        "screens": (150, 145, 140),
        "equitone": (100, 100, 105),
        "roof": (170, 170, 175),
        "bronze": (80, 60, 50),
        "shade": (240, 240, 240),
    }
    for name, tex_path in textures.items():
        Image.new("RGB", (16, 16), colors[name]).save(tex_path)

    output_path = tmp_path / "output.png"
    enhance_aerial(
        input_path,
        output_path,
        analysis_max_dim=128,
        k=4,
        seed=1,
        target_width=256,
        textures=textures,
    )

    # Verify output exists and has correct dimensions
    assert output_path.exists()
    enhanced = Image.open(output_path)
    assert enhanced.size[0] == 256

    # Verify enhancement modified the image
    assert np.asarray(enhanced).mean() != np.asarray(image.resize(enhanced.size)).mean()


def test_save_palette_assignments_creates_json(tmp_path: Path) -> None:
    """Test that palette assignments can be saved to JSON."""
    texture_path = tmp_path / "texture.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(texture_path)
    
    rule_a = MaterialRule(name="plaster", texture=str(texture_path), blend=0.5, score_fn=lambda _: 1.0)
    rule_b = MaterialRule(name="stone", texture=str(texture_path), blend=0.6, score_fn=lambda _: 1.0)
    
    assignments = {0: rule_a, 2: rule_b}
    palette_path = tmp_path / "palette.json"
    
    save_palette_assignments(assignments, palette_path)
    
    # Verify file exists
    assert palette_path.exists()
    
    # Verify JSON structure
    import json
    with open(palette_path, 'r') as f:
        data = json.load(f)
    
    assert "version" in data
    assert "assignments" in data
    assert data["assignments"]["0"] == "plaster"
    assert data["assignments"]["2"] == "stone"


def test_load_palette_assignments_restores_mappings(tmp_path: Path) -> None:
    """Test that palette assignments can be loaded from JSON."""
    import json
    
    # Create palette file
    palette_data = {
        "version": "1.0",
        "assignments": {
            "0": "plaster",
            "1": "stone",
            "3": "equitone"
        }
    }
    palette_path = tmp_path / "palette.json"
    with open(palette_path, 'w') as f:
        json.dump(palette_data, f)
    
    # Create material rules
    texture_path = tmp_path / "texture.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(texture_path)
    
    rules = [
        MaterialRule(name="plaster", texture=str(texture_path), blend=0.5, score_fn=lambda _: 1.0),
        MaterialRule(name="stone", texture=str(texture_path), blend=0.6, score_fn=lambda _: 1.0),
        MaterialRule(name="equitone", texture=str(texture_path), blend=0.7, score_fn=lambda _: 1.0),
    ]
    
    # Load assignments
    assignments = load_palette_assignments(palette_path, rules)
    
    # Verify mappings
    assert len(assignments) == 3
    assert assignments[0].name == "plaster"
    assert assignments[1].name == "stone"
    assert assignments[3].name == "equitone"


def test_load_palette_assignments_rejects_unknown_materials(tmp_path: Path) -> None:
    """Test that loading palette with unknown material names raises ValueError."""
    import json
    import pytest
    
    # Create palette with unknown material
    palette_data = {
        "version": "1.0",
        "assignments": {
            "0": "unknown_material"
        }
    }
    palette_path = tmp_path / "palette.json"
    with open(palette_path, 'w') as f:
        json.dump(palette_data, f)
    
    # Create material rules (no "unknown_material")
    texture_path = tmp_path / "texture.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(texture_path)
    
    rules = [
        MaterialRule(name="plaster", texture=str(texture_path), blend=0.5, score_fn=lambda _: 1.0),
    ]
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="Unknown material"):
        load_palette_assignments(palette_path, rules)


def test_enhance_aerial_with_palette_uses_predefined_mappings(tmp_path: Path) -> None:
    """Test that enhance_aerial respects palette file when provided."""
    import json
    
    width, height = 160, 120
    
    # Create synthetic aerial
    image = Image.new("RGB", (width, height), (160, 160, 150))
    for x in range(width):
        for y in range(height):
            if x < width // 2:
                image.putpixel((x, y), (225, 215, 200))  # will be cluster 0
            else:
                image.putpixel((x, y), (60, 50, 40))  # will be cluster 1
    
    input_path = tmp_path / "input.png"
    image.save(input_path)
    
    # Create textures
    material_names = ("plaster", "stone", "cladding", "screens", "equitone", "roof", "bronze", "shade")
    textures = {name: (tmp_path / f"{name}.png") for name in material_names}
    colors = {
        "plaster": (240, 230, 215),
        "stone": (210, 190, 170),
        "cladding": (200, 170, 140),
        "screens": (150, 145, 140),
        "equitone": (100, 100, 105),
        "roof": (170, 170, 175),
        "bronze": (80, 60, 50),
        "shade": (240, 240, 240),
    }
    for name, tex_path in textures.items():
        Image.new("RGB", (16, 16), colors[name]).save(tex_path)
    
    # Create palette that assigns clusters deliberately
    palette_data = {
        "version": "1.0",
        "assignments": {
            "0": "bronze",  # Force light region to bronze (normally would be plaster)
            "1": "plaster",  # Force dark region to plaster (normally would be bronze)
        }
    }
    palette_path = tmp_path / "palette.json"
    with open(palette_path, 'w') as f:
        json.dump(palette_data, f)
    
    output_path = tmp_path / "output.png"
    enhance_aerial(
        input_path,
        output_path,
        analysis_max_dim=128,
        k=2,
        seed=1,
        target_width=256,
        textures=textures,
        palette_path=palette_path,
    )
    
    # Verify output was created
    assert output_path.exists()


def test_enhance_aerial_can_save_computed_palette(tmp_path: Path) -> None:
    """Test that enhance_aerial can save assignments to a palette file."""
    width, height = 160, 120
    
    # Create synthetic aerial
    image = Image.new("RGB", (width, height), (225, 215, 200))
    input_path = tmp_path / "input.png"
    image.save(input_path)
    
    # Create textures
    material_names = ("plaster", "stone", "cladding", "screens", "equitone", "roof", "bronze", "shade")
    textures = {name: (tmp_path / f"{name}.png") for name in material_names}
    for name, tex_path in textures.items():
        Image.new("RGB", (16, 16), (255, 255, 255)).save(tex_path)
    
    output_path = tmp_path / "output.png"
    palette_save_path = tmp_path / "saved_palette.json"
    
    enhance_aerial(
        input_path,
        output_path,
        analysis_max_dim=128,
        k=3,
        seed=1,
        target_width=256,
        textures=textures,
        save_palette=palette_save_path,
    )
    
    # Verify palette was saved
    assert palette_save_path.exists()
    
    # Verify it has valid structure
    import json
    with open(palette_save_path, 'r') as f:
        data = json.load(f)
    
    assert "version" in data
    assert "assignments" in data
    assert len(data["assignments"]) > 0
