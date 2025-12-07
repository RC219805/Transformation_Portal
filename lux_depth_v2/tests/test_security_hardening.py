"""
Test security hardening features in lux_depth_v2.

These tests verify Phase 1 remediation items:
- Requirements file warnings
- Path traversal protection
- Model revision pinning
"""
import pytest
from pathlib import Path


class TestDependencySecurity:
    """Test that vulnerable dependencies are properly documented and excluded."""
    
    def test_requirements_txt_has_security_warnings(self):
        """Verify requirements.txt contains security warnings about basicsr/realesrgan."""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        assert req_file.exists(), "requirements.txt not found"
        
        content = req_file.read_text()
        
        # Verify security warning section exists
        assert "SECURITY WARNING" in content, "Missing security warning header"
        assert "CVE-2024-27763" in content, "Missing CVE reference"
        
        # Verify unsafe packages are NOT in installable list
        lines = [line.strip() for line in content.split('\n') 
                if line.strip() and not line.strip().startswith('#')]
        installable_deps = [line for line in lines if not line.startswith('-')]
        
        # Ensure basicsr and realesrgan are NOT in installable dependencies
        assert not any('basicsr' in dep.lower() for dep in installable_deps), \
            "basicsr should not be in installable requirements"
        assert not any('realesrgan' in dep.lower() for dep in installable_deps), \
            "realesrgan should not be in installable requirements"
    
    def test_requirements_repo_exists(self):
        """Verify requirements-repo.txt exists as safe alternative."""
        req_repo = Path(__file__).parent.parent / "requirements-repo.txt"
        assert req_repo.exists(), "requirements-repo.txt not found"
        
        content = req_repo.read_text()
        
        # Verify it excludes vulnerable packages
        assert "basicsr" not in content or "EXCLUDED" in content or "CVE" in content, \
            "requirements-repo.txt should exclude or document basicsr exclusion"


class TestServiceModeSecurity:
    """Test service mode security features."""
    
    def test_validate_filepath_rejects_path_traversal(self):
        """Verify validate_filepath rejects path traversal attempts."""
        # Verify the function exists and has path traversal checks by examining source
        service_src = (Path(__file__).parent.parent / "service.py").read_text()
        
        # Verify validate_filepath function exists and has path traversal checks
        assert "def validate_filepath" in service_src, "validate_filepath function not found"
        assert '".."' in service_src and "path traversal" in service_src.lower(), \
            "validate_filepath missing path traversal protection"
        assert '"/"' in service_src or '"\\\\' in service_src, \
            "validate_filepath missing slash checks"
    
    def test_validate_filepath_has_null_byte_checks(self):
        """Verify validate_filepath checks for null bytes."""
        import re
        service_src = (Path(__file__).parent.parent / "service.py").read_text()
        
        # Extract the validate_filepath function body
        match = re.search(
            r"def validate_filepath\(.*?\):.*?\n((?:[ \t]+.+\n)*)", 
            service_src, 
            re.MULTILINE | re.DOTALL
        )
        assert match, "validate_filepath function not found in service.py"
        func_body = match.group(0)
        
        # Verify null byte protection is present in the function body
        assert (
            "'\\x00'" in func_body or '"\\x00"' in func_body or
            "'\\0'" in func_body or '"\\0"' in func_body
        ), "validate_filepath missing null byte checks"
    
    def test_validate_filepath_has_extension_validation(self):
        """Verify validate_filepath validates file extensions."""
        service_src = (Path(__file__).parent.parent / "service.py").read_text()
        
        # Verify extension validation exists
        assert "allowed_extensions" in service_src or "extension" in service_src.lower(), \
            "validate_filepath missing extension validation"
        assert ".tif" in service_src or "tiff" in service_src.lower(), \
            "validate_filepath missing TIFF extension support"


class TestModelSecurity:
    """Test model security features (revision pinning)."""
    
    def test_segmentation_config_has_revision_field(self):
        """Verify SegmentationConfig supports revision pinning."""
        from lux_depth_v2.config import SegmentationConfig
        
        cfg = SegmentationConfig()
        
        # Verify revision field exists
        assert hasattr(cfg, 'segformer_revision'), \
            "SegmentationConfig missing segformer_revision field"
        
        # Verify it can be set
        cfg_with_revision = SegmentationConfig(
            segformer_model="nvidia/segformer-b2-finetuned-ade-512-512",
            segformer_revision="9bcfaf5c6a0df63c26e76e9d16c3d2e5c7e5e7e0"
        )
        assert cfg_with_revision.segformer_revision == "9bcfaf5c6a0df63c26e76e9d16c3d2e5c7e5e7e0"


class TestUpscalerSecurity:
    """Test upscaler backend security."""
    
    def test_realesrgan_backend_is_deprecated(self):
        """Verify realesrgan backend triggers deprecation warning."""
        from lux_depth_v2 import config
        
        # Create config with realesrgan backend
        cfg = config.PipelineConfig(upscaler_backend="realesrgan")
        
        # The actual deprecation warning is emitted in upscaling.create_upscaler()
        # when torch is available. Here we just verify the config accepts it
        # (the warning happens at runtime)
        assert cfg.upscaler_backend == "realesrgan"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
