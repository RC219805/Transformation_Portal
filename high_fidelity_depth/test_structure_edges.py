#!/usr/bin/env python3
"""Tests for structure-aware edge extraction."""

import numpy as np
import cv2
import pytest
from high_fidelity_depth.quality_metrics import extract_structure_edges, classify_scene_type


def test_bilateral_suppresses_texture():
    """Verify bilateral filter removes texture while preserving structure."""
    # Create synthetic image: strong edge + noise texture
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, 50:] = 200  # Strong vertical edge

    # Add noise (simulating texture)
    noise = np.random.randint(-20, 20, (100, 100), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Extract edges: raw vs structure
    raw_edges = cv2.Canny(image, 50, 150)
    structure_edges = extract_structure_edges(image)

    # Structure edges should have fewer pixels (texture suppressed)
    raw_count = np.count_nonzero(raw_edges)
    structure_count = np.count_nonzero(structure_edges)

    assert structure_count < raw_count, f"Structure edges ({structure_count}) should be < raw edges ({raw_count})"

    # Verify strong edge is preserved (check middle column region)
    assert structure_edges[:, 48:52].sum() > 0, "Strong edge should be preserved"


def test_scene_classification():
    """Verify texture vs structure classification."""
    # Texture-dominated: many raw edges, few structure edges
    raw_edges_texture = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    structure_edges_texture = np.zeros((100, 100), dtype=np.uint8)
    structure_edges_texture[50, :] = 255  # One horizontal line

    scene_type = classify_scene_type(raw_edges_texture, structure_edges_texture)
    assert scene_type == "texture_dominated"

    # Structure-dominated: similar raw and structure edge counts
    raw_edges_struct = np.zeros((100, 100), dtype=np.uint8)
    raw_edges_struct[50, :] = 255
    raw_edges_struct[:, 50] = 255
    structure_edges_struct = raw_edges_struct.copy()

    scene_type = classify_scene_type(raw_edges_struct, structure_edges_struct)
    assert scene_type == "structure_dominated"


def test_structure_edges_grayscale_input():
    """Test extract_structure_edges with grayscale input."""
    # Create grayscale image with texture
    image = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
    image[:, 50:] += 50  # Add edge

    edges = extract_structure_edges(image)

    # Should return binary edge map
    assert edges.shape == (100, 100)
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)).issubset({0, 255})


def test_structure_edges_rgb_input():
    """Test extract_structure_edges with RGB input."""
    # Create RGB image
    image = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
    image[:, 50:, :] += 50  # Add edge

    edges = extract_structure_edges(image)

    # Should return binary edge map
    assert edges.shape == (100, 100)
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)).issubset({0, 255})


def test_classify_scene_type_edge_cases():
    """Test scene classification edge cases."""
    # Zero structure edges → texture dominated
    raw_edges = np.ones((100, 100), dtype=np.uint8) * 255
    structure_edges = np.zeros((100, 100), dtype=np.uint8)

    scene_type = classify_scene_type(raw_edges, structure_edges)
    assert scene_type == "texture_dominated"

    # Equal counts → structure dominated
    edges = np.zeros((100, 100), dtype=np.uint8)
    edges[50, :] = 255

    scene_type = classify_scene_type(edges, edges)
    assert scene_type == "structure_dominated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
