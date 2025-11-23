"""
Tests for BasicSR-TP Vendored Package
======================================

Validates that the security-hardened vendored BasicSR-TP package:
1. Imports correctly
2. RRDBNet architecture is functional
3. No vulnerable SLURM code is included
4. API is compatible with original BasicSR

Security Advisory: CVE-2024-27763
- Tests ensure vulnerable code is NOT present
- Tests verify secure alternative works correctly
"""

import pytest
import sys
from pathlib import Path


class TestBasicSRTPImports:
    """Test that BasicSR-TP imports work correctly."""

    def test_import_from_archs(self):
        """Test importing RRDBNet from basicsr_tp.archs."""
        from basicsr_tp.archs.rrdbnet_arch import RRDBNet
        assert RRDBNet is not None

    def test_import_from_package_level(self):
        """Test importing RRDBNet from package level."""
        from basicsr_tp import RRDBNet
        assert RRDBNet is not None

    def test_package_version(self):
        """Test package has correct version."""
        import basicsr_tp
        assert hasattr(basicsr_tp, '__version__')
        assert 'tp' in basicsr_tp.__version__  # TP patch version
        assert '1.4.2' in basicsr_tp.__version__  # Based on BasicSR 1.4.2


class TestRRDBNetArchitecture:
    """Test RRDBNet architecture functionality."""

    @pytest.fixture
    def model(self):
        """Create a basic RRDBNet model for testing."""
        from basicsr_tp import RRDBNet
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

    def test_model_instantiation(self, model):
        """Test that RRDBNet can be instantiated."""
        assert model is not None
        from torch import nn
        assert isinstance(model, nn.Module)

    def test_model_parameters(self, model):
        """Test that model has trainable parameters."""
        params = list(model.parameters())
        assert len(params) > 0
        # RRDBNet should have millions of parameters
        total_params = sum(p.numel() for p in params)
        assert total_params > 1_000_000  # At least 1M parameters

    def test_forward_pass(self, model):
        """Test forward pass with dummy input."""
        import torch
        # Create dummy input: batch_size=1, channels=3, height=64, width=64
        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            y = model(x)

        # Output should be 4x upsampled (64 -> 256)
        assert y.shape == torch.Size([1, 3, 256, 256])

    def test_forward_pass_different_sizes(self, model):
        """Test forward pass with various input sizes."""
        import torch

        test_sizes = [
            (32, 32),   # Small
            (64, 64),   # Medium
            (128, 128), # Large
        ]

        for h, w in test_sizes:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                y = model(x)
            # Should upscale by 4x
            assert y.shape == torch.Size([1, 3, h * 4, w * 4])


class TestSecurityValidation:
    """Security tests to ensure vulnerable code is NOT present."""

    def test_no_dist_util_module(self):
        """Ensure dist_util.py (vulnerable file) is not present."""
        import basicsr_tp
        assert not hasattr(basicsr_tp, 'utils')

        # Try to import dist_util - should fail
        with pytest.raises(ImportError):
            from basicsr_tp.utils.dist_util import init_dist  # noqa: F401

    def test_no_slurm_code_in_files(self):
        """Ensure no SLURM-related executable code exists in vendored files."""
        basicsr_tp_dir = Path(__file__).parent.parent / "basicsr_tp"
        assert basicsr_tp_dir.exists()

        # Search all Python files for actual executable SLURM code
        for py_file in basicsr_tp_dir.rglob("*.py"):
            # Skip __init__.py and README files which contain security documentation
            if py_file.name in ['__init__.py', 'README.md']:
                continue

            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                # Check for actual vulnerable code patterns (not in comments/docstrings)
                in_docstring = False
                for i, line in enumerate(lines, start=1):
                    stripped = line.strip()

                    # Track docstring state
                    if '"""' in stripped or "'''" in stripped:
                        in_docstring = not in_docstring
                        continue

                    # Skip comments and docstrings
                    if in_docstring or stripped.startswith('#'):
                        continue

                    # Check for actual vulnerable code in executable lines
                    forbidden_patterns = [
                        'scontrol show hostname',  # The actual vulnerable command
                        'subprocess.getoutput',    # Shell execution method
                        'os.environ[\'SLURM',      # Accessing SLURM environment
                        'def init_dist',           # Distributed init function
                        'def _init_dist_slurm',    # SLURM-specific init
                    ]

                    for pattern in forbidden_patterns:
                        if pattern in line and not line.strip().startswith('#'):
                            pytest.fail(
                                f"Found vulnerable SLURM code in {py_file}:{i}\n"
                                f"  Pattern: {pattern}\n"
                                f"  Line: {line.strip()}"
                            )

    def test_no_subprocess_calls(self):
        """Ensure no subprocess calls exist (vulnerability vector)."""
        basicsr_tp_dir = Path(__file__).parent.parent / "basicsr_tp"

        for py_file in basicsr_tp_dir.rglob("*.py"):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for subprocess imports/calls
                forbidden = ['subprocess.', 'import subprocess', 'from subprocess']
                for pattern in forbidden:
                    assert pattern not in content, \
                        f"Found subprocess usage in {py_file} - potential security risk"


class TestAPICompatibility:
    """Test that API is compatible with original BasicSR."""

    def test_rrdbnet_signature_matches(self):
        """Test that RRDBNet has the same constructor signature as original."""
        from basicsr_tp import RRDBNet
        import inspect

        sig = inspect.signature(RRDBNet.__init__)
        params = list(sig.parameters.keys())

        # Expected parameters from original BasicSR
        expected = ['self', 'num_in_ch', 'num_out_ch', 'scale', 'num_feat', 'num_block', 'num_grow_ch']
        assert params == expected

    def test_helper_functions_exist(self):
        """Test that required helper functions are available."""
        from basicsr_tp.archs.rrdbnet_arch import (
            default_init_weights,
            make_layer,
            pixel_unshuffle
        )
        assert callable(default_init_weights)
        assert callable(make_layer)
        assert callable(pixel_unshuffle)

    def test_residual_blocks_exist(self):
        """Test that ResidualDenseBlock and RRDB classes exist."""
        from basicsr_tp.archs.rrdbnet_arch import ResidualDenseBlock, RRDB
        assert ResidualDenseBlock is not None
        assert RRDB is not None


class TestDocumentation:
    """Test that security documentation is present."""

    def test_security_note_in_module(self):
        """Test that security advisory is in module docstring."""
        import basicsr_tp.archs.rrdbnet_arch as module
        docstring = module.__doc__
        assert docstring is not None
        assert 'CVE-2024-27763' in docstring
        assert 'security' in docstring.lower()

    def test_readme_exists(self):
        """Test that README with security info exists."""
        readme = Path(__file__).parent.parent / "basicsr_tp" / "README.md"
        assert readme.exists()

        content = readme.read_text()
        assert 'CVE-2024-27763' in content
        assert 'Security' in content
        assert 'SLURM' in content

    def test_package_init_has_metadata(self):
        """Test that package __init__ has version and license."""
        import basicsr_tp
        assert hasattr(basicsr_tp, '__version__')
        assert hasattr(basicsr_tp, '__license__')
        assert basicsr_tp.__license__ == 'Apache-2.0'


@pytest.mark.skipif(
    'torch' not in sys.modules and not pytest.importorskip("torch", reason="PyTorch not available"),
    reason="PyTorch not installed"
)
class TestRealESRGANIntegration:
    """Test integration with Real-ESRGAN (if available)."""

    def test_realesrgan_can_use_vendored_rrdbnet(self):
        """Test that Real-ESRGAN can use our vendored RRDBNet."""
        try:
            from realesrgan import RealESRGANer
            from basicsr_tp import RRDBNet

            # Create model
            model = RRDBNet(num_in_ch=3, num_out_ch=3)

            # RealESRGANer should accept our model
            # Note: This will fail without weights file, but constructor should work
            upsampler = RealESRGANer(
                scale=4,
                model_path=None,  # Don't load weights
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False
            )
            assert upsampler is not None

        except ImportError:
            pytest.skip("Real-ESRGAN not installed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
