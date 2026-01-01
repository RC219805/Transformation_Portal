#!/usr/bin/env python3
"""
Tests for Pre-Commit Security Check Script
==========================================

Validates that security checks correctly identify and block:
- Sensitive files
- Bidirectional Unicode characters
- Large files
- Output artifacts
"""

import sys
import tempfile
from pathlib import Path

# Add scripts/security to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts' / 'security'))

from pre_commit_security_check import (
    check_bidi_unicode,
    check_sensitive_file,
    check_output_directory,
    check_file_size,
    SecurityViolation,
    BIDI_CHARS
)


class TestBidiUnicodeDetection:
    """Test bidirectional Unicode character detection."""
    
    def test_clean_python_file(self, tmp_path):
        """Test that clean Python files pass."""
        test_file = tmp_path / "clean.py"
        test_file.write_text("def hello():\n    print('Hello, World!')\n")
        
        violations = check_bidi_unicode(test_file)
        assert len(violations) == 0
    
    def test_python_with_bidi_lro(self, tmp_path):
        """Test detection of LEFT-TO-RIGHT OVERRIDE."""
        test_file = tmp_path / "malicious.py"
        # Include LRO character
        test_file.write_text(f"def safe():\n    print('Safe{chr(0x202E)}Malicious')\n")
        
        violations = check_bidi_unicode(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'BIDI_UNICODE'
        assert 'RIGHT-TO-LEFT OVERRIDE' in violations[0].details
    
    def test_markdown_ignored(self, tmp_path):
        """Test that markdown files are ignored for bidi check."""
        test_file = tmp_path / "doc.md"
        test_file.write_text(f"# Title\n\nSome text{chr(0x202E)} with bidi\n")
        
        violations = check_bidi_unicode(test_file)
        # Markdown is not in CODE_EXTENSIONS, so should be ignored
        assert len(violations) == 0
    
    def test_shell_script_with_bidi(self, tmp_path):
        """Test detection in shell scripts."""
        test_file = tmp_path / "script.sh"
        test_file.write_text(f"#!/bin/bash\necho 'test{chr(0x202A)}malicious'\n")
        
        violations = check_bidi_unicode(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'BIDI_UNICODE'


class TestSensitiveFileDetection:
    """Test sensitive file pattern detection."""
    
    def test_bash_history(self, tmp_path):
        """Test detection of .bash_history."""
        test_file = tmp_path / ".bash_history"
        test_file.write_text("ls -la\n")
        
        violations = check_sensitive_file(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'SENSITIVE_FILE'
    
    def test_pem_file(self, tmp_path):
        """Test detection of .pem credential file."""
        test_file = tmp_path / "server.pem"
        test_file.write_text("-----BEGIN CERTIFICATE-----\n")
        
        violations = check_sensitive_file(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'SENSITIVE_FILE'
    
    def test_ssh_key(self, tmp_path):
        """Test detection of SSH private key."""
        test_file = tmp_path / "id_rsa"
        test_file.write_text("-----BEGIN RSA PRIVATE KEY-----\n")
        
        violations = check_sensitive_file(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'SENSITIVE_FILE'
    
    def test_env_file(self, tmp_path):
        """Test detection of .env file."""
        test_file = tmp_path / ".env"
        test_file.write_text("API_KEY=secret123\n")
        
        violations = check_sensitive_file(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'SENSITIVE_FILE'
    
    def test_pkg_info(self, tmp_path):
        """Test detection of PKG-INFO build artifact."""
        test_file = tmp_path / "PKG-INFO"
        test_file.write_text("Metadata-Version: 2.1\n")
        
        violations = check_sensitive_file(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'SENSITIVE_FILE'
    
    def test_normal_python_file(self, tmp_path):
        """Test that normal Python files don't trigger false positives."""
        test_file = tmp_path / "main.py"
        test_file.write_text("import os\nprint('Hello')\n")
        
        violations = check_sensitive_file(test_file)
        assert len(violations) == 0


class TestOutputDirectoryDetection:
    """Test output directory pattern detection."""
    
    def test_output_directory(self, tmp_path):
        """Test detection of files in output_* directories."""
        test_dir = tmp_path / "output_test_20260101"
        test_dir.mkdir()
        test_file = test_dir / "result.png"
        test_file.write_text("fake image")
        
        violations = check_output_directory(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'OUTPUT_ARTIFACT'
    
    def test_phase_outputs(self, tmp_path):
        """Test detection of phase task outputs."""
        test_dir = tmp_path / "phase2_task1_outputs"
        test_dir.mkdir()
        test_file = test_dir / "result.json"
        test_file.write_text('{"test": true}')
        
        violations = check_output_directory(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'OUTPUT_ARTIFACT'
    
    def test_normal_directory(self, tmp_path):
        """Test that normal directories don't trigger false positives."""
        test_dir = tmp_path / "src" / "module"
        test_dir.mkdir(parents=True)
        test_file = test_dir / "code.py"
        test_file.write_text("def func(): pass")
        
        violations = check_output_directory(test_file)
        assert len(violations) == 0


class TestFileSizeDetection:
    """Test large file detection."""
    
    def test_small_file(self, tmp_path):
        """Test that small files pass."""
        test_file = tmp_path / "small.txt"
        test_file.write_text("Small content\n" * 100)
        
        violations = check_file_size(test_file)
        assert len(violations) == 0
    
    def test_large_file(self, tmp_path):
        """Test detection of files >5MB."""
        test_file = tmp_path / "large.bin"
        # Create a file larger than 5MB
        with open(test_file, 'wb') as f:
            f.write(b'0' * (6 * 1024 * 1024))  # 6MB
        
        violations = check_file_size(test_file)
        assert len(violations) > 0
        assert violations[0].violation_type == 'LARGE_FILE'
        assert 'Git LFS' in violations[0].details


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_multiple_violations(self, tmp_path):
        """Test that multiple violations are detected."""
        # Create a .bash_history file
        history = tmp_path / ".bash_history"
        history.write_text("export API_KEY=secret\n")
        
        violations = []
        violations.extend(check_sensitive_file(history))
        
        assert len(violations) > 0
        assert all(v.violation_type == 'SENSITIVE_FILE' for v in violations)
    
    def test_clean_repository_structure(self, tmp_path):
        """Test that normal repository files pass all checks."""
        # Create typical repository files
        files = [
            tmp_path / "README.md",
            tmp_path / "src" / "main.py",
            tmp_path / "tests" / "test_main.py",
            tmp_path / "docs" / "guide.md",
        ]
        
        for f in files:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("# Content\n")
        
        all_violations = []
        for f in files:
            all_violations.extend(check_sensitive_file(f))
            all_violations.extend(check_output_directory(f))
            all_violations.extend(check_file_size(f))
            all_violations.extend(check_bidi_unicode(f))
        
        assert len(all_violations) == 0


def run_tests():
    """Run all tests manually (for environments without pytest)."""
    import traceback
    
    test_classes = [
        TestBidiUnicodeDetection,
        TestSensitiveFileDetection,
        TestOutputDirectoryDetection,
        TestFileSizeDetection,
        TestIntegration,
    ]
    
    total_tests = 0
    passed = 0
    failed = 0
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        for test_class in test_classes:
            print(f"\n{test_class.__name__}:")
            test_instance = test_class()
            
            for method_name in dir(test_instance):
                if method_name.startswith('test_'):
                    total_tests += 1
                    try:
                        method = getattr(test_instance, method_name)
                        method(tmp_path)
                        print(f"  ✓ {method_name}")
                        passed += 1
                    except AssertionError as e:
                        print(f"  ✗ {method_name}: {e}")
                        failed += 1
                    except Exception as e:
                        print(f"  ✗ {method_name}: {e}")
                        traceback.print_exc()
                        failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Total: {total_tests} | Passed: {passed} | Failed: {failed}")
    print(f"{'=' * 60}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    # Can run with pytest or standalone
    try:
        import pytest
        sys.exit(pytest.main([__file__, '-v']))
    except ImportError:
        sys.exit(run_tests())
