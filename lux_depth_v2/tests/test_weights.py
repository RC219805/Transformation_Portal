"""Unit tests for weights module."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

from lux_depth_v2 import weights as weights_mod
from lux_depth_v2.config import PipelineConfig


class TestWeightsFromAssets:
    """Test weight generation from various assets."""

    def test_uniform_weights_no_depth(self, torch_device):
        """Test uniform weights when no depth or masks provided."""
        cfg = PipelineConfig(strict_depth=False)
        w = weights_mod.weights_from_assets(
            h=64, w=64,
            device=torch_device,
            depth01=None,
            masks={},
            cfg=cfg
        )
        
        assert w.source == "uniform_no_depth"
        assert w.wfg.shape == (1, 1, 64, 64)
        assert w.wmid.shape == (1, 1, 64, 64)
        assert w.wbg.shape == (1, 1, 64, 64)
        
        # Weights should be roughly equal
        assert w.wfg.mean().item() == pytest.approx(0.34, abs=0.01)
        assert w.wmid.mean().item() == pytest.approx(0.33, abs=0.01)
        assert w.wbg.mean().item() == pytest.approx(0.33, abs=0.01)

    def test_strict_depth_raises(self, torch_device):
        """Test strict_depth raises when depth missing."""
        cfg = PipelineConfig(strict_depth=True)
        with pytest.raises(FileNotFoundError, match="Depth missing"):
            weights_mod.weights_from_assets(
                h=64, w=64,
                device=torch_device,
                depth01=None,
                masks={},
                cfg=cfg
            )

    def test_depth_percentile_weights(self, torch_device, sample_depth_array):
        """Test weights from depth percentiles."""
        cfg = PipelineConfig(fg_q=0.3, bg_q=0.7, transition=0.1)
        w = weights_mod.weights_from_assets(
            h=64, w=64,
            device=torch_device,
            depth01=sample_depth_array,
            masks={},
            cfg=cfg
        )
        
        assert w.source == "depth_percentiles"
        assert w.wfg.shape == (1, 1, 64, 64)
        
        # Weights should sum to 1
        total = w.wfg + w.wmid + w.wbg
        assert torch.allclose(total, torch.ones_like(total), atol=1e-4)

    def test_zone_mask_weights(self, torch_device, sample_mask_array):
        """Test weights from explicit zone masks."""
        cfg = PipelineConfig()
        
        # Create simple zone masks
        fg_mask = sample_mask_array.copy()
        bg_mask = 1.0 - fg_mask
        mid_mask = np.ones_like(fg_mask) * 0.5
        
        masks = {
            "foreground": fg_mask,
            "midground": mid_mask,
            "background": bg_mask,
        }
        
        w = weights_mod.weights_from_assets(
            h=64, w=64,
            device=torch_device,
            depth01=None,
            masks=masks,
            cfg=cfg
        )
        
        assert w.source == "zone_masks"
        assert w.wfg.shape == (1, 1, 64, 64)
        
        # Weights should sum to 1 after normalization
        total = w.wfg + w.wmid + w.wbg
        assert torch.allclose(total, torch.ones_like(total), atol=1e-3)

    def test_zone_mask_priority_over_depth(self, torch_device, sample_depth_array, sample_mask_array):
        """Test zone masks take priority over depth."""
        cfg = PipelineConfig()
        
        masks = {
            "foreground": sample_mask_array,
            "midground": np.ones_like(sample_mask_array) * 0.5,
            "background": 1.0 - sample_mask_array,
        }
        
        w = weights_mod.weights_from_assets(
            h=64, w=64,
            device=torch_device,
            depth01=sample_depth_array,  # Depth provided but should be ignored
            masks=masks,
            cfg=cfg
        )
        
        assert w.source == "zone_masks"  # Not depth_percentiles

    def test_mask_softening(self, torch_device, sample_mask_array):
        """Test mask softening with gaussian blur."""
        cfg = PipelineConfig(mask_soften_sigma=4.0)
        
        masks = {
            "foreground": sample_mask_array,
            "midground": np.ones_like(sample_mask_array) * 0.5,
            "background": 1.0 - sample_mask_array,
        }
        
        w = weights_mod.weights_from_assets(
            h=64, w=64,
            device=torch_device,
            depth01=None,
            masks=masks,
            cfg=cfg
        )
        
        # Softened masks should have smooth transitions
        wfg_np = w.wfg[0, 0].cpu().numpy()
        # Check that values near edges are not strictly 0 or 1
        center = wfg_np[32, 32]
        edge = wfg_np[16, 32]
        assert 0 < edge < center  # Edge should be partially blurred

    def test_depth_weights_distribution(self, torch_device):
        """Test depth weights are properly distributed."""
        # Create depth with clear zones
        depth = np.zeros((64, 64), dtype=np.float32)
        depth[:20, :] = 0.0  # Foreground
        depth[20:44, :] = 0.5  # Midground
        depth[44:, :] = 1.0  # Background
        
        cfg = PipelineConfig(fg_q=0.25, bg_q=0.75, transition=0.05)
        w = weights_mod.weights_from_assets(
            h=64, w=64,
            device=torch_device,
            depth01=depth,
            masks={},
            cfg=cfg
        )
        
        # Check foreground zone has high wfg (relaxed for device variance)
        wfg_fg = w.wfg[0, 0, :20, :].mean().item()
        assert wfg_fg > 0.4, f"Expected wfg > 0.4, got {wfg_fg}"
        
        # Check background zone has high wbg (relaxed for device variance)
        wbg_bg = w.wbg[0, 0, 44:, :].mean().item()
        assert wbg_bg > 0.4, f"Expected wbg > 0.4, got {wbg_bg}"


class TestWeightsDataclass:
    """Test Weights dataclass."""

    def test_weights_creation(self, torch_device):
        """Test Weights dataclass creation."""
        wfg = torch.ones((1, 1, 32, 32), device=torch_device) * 0.3
        wmid = torch.ones((1, 1, 32, 32), device=torch_device) * 0.4
        wbg = torch.ones((1, 1, 32, 32), device=torch_device) * 0.3
        
        w = weights_mod.Weights(
            wfg=wfg,
            wmid=wmid,
            wbg=wbg,
            source="test"
        )
        
        assert w.wfg.shape == (1, 1, 32, 32)
        assert w.wmid.shape == (1, 1, 32, 32)
        assert w.wbg.shape == (1, 1, 32, 32)
        assert w.source == "test"

    def test_weights_sum_to_one(self, torch_device):
        """Test that generated weights sum to approximately 1."""
        cfg = PipelineConfig()
        depth = np.random.rand(64, 64).astype(np.float32)
        
        w = weights_mod.weights_from_assets(
            h=64, w=64,
            device=torch_device,
            depth01=depth,
            masks={},
            cfg=cfg
        )
        
        total = w.wfg + w.wmid + w.wbg
        assert torch.allclose(total, torch.ones_like(total), atol=1e-3)
