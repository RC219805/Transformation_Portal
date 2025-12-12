"""Tests for Phase 2 CLIP Material Classifier."""

import pytest
import torch

from lux_depth_v2.materials_v2 import CLIPMaterialClassifier


@pytest.fixture
def device():
    """Get torch device for testing."""
    return torch.device('cpu')


@pytest.fixture
def clip_classifier(device):
    """Create CLIP classifier instance."""
    return CLIPMaterialClassifier(device, model_name='ViT-B/32')


def test_clip_initialization(clip_classifier):
    """Test CLIP classifier initializes correctly."""
    assert clip_classifier is not None
    assert clip_classifier.model_name == 'ViT-B/32'
    assert len(clip_classifier.material_templates) == 28  # 8 Phase 1 + 20 Phase 2
    assert len(clip_classifier.material_embeddings) == 28


def test_clip_material_templates(clip_classifier):
    """Test material templates are comprehensive."""
    # Phase 1 materials
    assert 'wood' in clip_classifier.material_templates
    assert 'metal' in clip_classifier.material_templates
    assert 'glass' in clip_classifier.material_templates
    assert 'water' in clip_classifier.material_templates
    
    # Phase 2 expanded materials
    assert 'pool_water_surface' in clip_classifier.material_templates
    assert 'stone_paver' in clip_classifier.material_templates
    assert 'stucco_wall' in clip_classifier.material_templates
    assert 'sky_gradient' in clip_classifier.material_templates
    
    # Each material should have multiple templates
    for material, templates in clip_classifier.material_templates.items():
        assert isinstance(templates, list)
        assert len(templates) >= 3, f"{material} should have at least 3 templates"


def test_clip_classify_image(clip_classifier, device):
    """Test CLIP zero-shot classification."""
    # Create dummy RGB image
    rgb = torch.rand(1, 3, 512, 512, device=device)
    
    # Classify
    scores = clip_classifier.classify_image(rgb)
    
    # Verify output
    assert isinstance(scores, dict)
    assert len(scores) == 28  # All material classes
    
    # All scores should be in [0, 1]
    for material, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{material} score {score} out of range"


def test_clip_classify_subset(clip_classifier, device):
    """Test CLIP classification with material subset."""
    rgb = torch.rand(1, 3, 512, 512, device=device)
    
    # Classify only specific materials
    subset = ['wood', 'metal', 'glass']
    scores = clip_classifier.classify_image(rgb, material_classes=subset)
    
    # Verify only requested materials are scored
    assert set(scores.keys()) == set(subset)


def test_clip_natural_language_query(clip_classifier, device):
    """Test natural language query interface."""
    rgb = torch.rand(1, 3, 512, 512, device=device)
    
    # Query for reflective surfaces
    mask = clip_classifier.query_natural_language(rgb, "surfaces that would reflect light")
    
    # Verify output shape and range
    assert mask.shape == (1, 1, 512, 512)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_clip_hybrid_fusion(clip_classifier, device):
    """Test hybrid SegFormer+CLIP fusion."""
    rgb = torch.rand(1, 3, 512, 512, device=device)
    
    # Create mock SegFormer outputs
    segformer_masks = {
        'wood': torch.rand(1, 1, 512, 512, device=device),
        'metal': torch.rand(1, 1, 512, 512, device=device),
    }
    segformer_confidences = {
        'wood': torch.full((1, 1, 512, 512), 0.8, device=device),
        'metal': torch.full((1, 1, 512, 512), 0.6, device=device),
    }
    
    # Fuse
    refined_masks = clip_classifier.fuse_with_segformer(
        rgb, segformer_masks, segformer_confidences
    )
    
    # Verify output
    assert isinstance(refined_masks, dict)
    assert len(refined_masks) >= 2
    
    # All masks should be valid
    for material, mask in refined_masks.items():
        assert mask.shape == (1, 1, 512, 512)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


def test_clip_embeddings_cached(clip_classifier):
    """Test that material embeddings are precomputed."""
    # Embeddings should be precomputed during init
    assert hasattr(clip_classifier, 'material_embeddings')
    assert len(clip_classifier.material_embeddings) == 28  # All classes
    
    # Each embedding should be a tensor
    for material, embedding in clip_classifier.material_embeddings.items():
        assert isinstance(embedding, torch.Tensor)
        assert embedding.ndim == 1  # 1D embedding vector


@pytest.mark.parametrize("material,expected_templates", [
    ('pool_water_surface', ['swimming pool water', 'reflective water']),
    ('wood', ['wood', 'wooden']),
    ('sky_gradient', ['sky']),
])
def test_clip_template_keywords(clip_classifier, material, expected_templates):
    """Test that material templates contain expected keywords."""
    templates = clip_classifier.material_templates[material]
    templates_str = ' '.join(templates).lower()
    
    for keyword in expected_templates:
        assert keyword in templates_str, f"'{keyword}' not found in {material} templates"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
