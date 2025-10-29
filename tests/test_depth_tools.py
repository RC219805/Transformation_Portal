#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for depth_tools.py batch processing and error handling
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Import from the module location
from src.transformation_portal.pipelines.depth_tools import (
    BatchOptions,
    process_batch,
    main,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = Path(tmpdir) / "images"
        depths_dir = Path(tmpdir) / "depths"
        out_dir = Path(tmpdir) / "output"
        
        images_dir.mkdir()
        depths_dir.mkdir()
        out_dir.mkdir()
        
        yield {
            "images": str(images_dir),
            "depths": str(depths_dir),
            "output": str(out_dir),
        }


@pytest.fixture
def sample_image():
    """Create a sample RGB image as numpy array"""
    return np.random.rand(100, 100, 3).astype(np.float32)


@pytest.fixture
def sample_depth():
    """Create a sample depth map as numpy array"""
    return np.random.rand(100, 100).astype(np.float32) * 65535


def create_test_files(temp_dirs, num_images=3, create_depth_for_all=True):
    """
    Helper to create test image and depth files
    
    Args:
        temp_dirs: fixture with directory paths
        num_images: number of test images to create
        create_depth_for_all: if False, skip creating depth for last image
    
    Returns:
        list of base names created
    """
    bases = []
    for i in range(num_images):
        base = f"test_image_{i:03d}"
        bases.append(base)
        
        # Create source image
        img_arr = np.random.rand(100, 100, 3) * 255
        img = Image.fromarray(img_arr.astype(np.uint8))
        img_path = Path(temp_dirs["images"]) / f"{base}.png"
        img.save(img_path)
        
        # Create depth map (skip last one if requested)
        if create_depth_for_all or i < num_images - 1:
            depth_arr = np.random.rand(100, 100) * 65535
            depth = Image.fromarray(depth_arr.astype(np.uint16))
            depth_path = Path(temp_dirs["depths"]) / f"{base}_depth16.png"
            depth.save(depth_path)
    
    return bases


class TestBatchProcessing:
    """Test batch processing functionality"""
    
    def test_successful_batch_processing(self, temp_dirs):
        """Test that batch processing completes successfully with all valid files"""
        create_test_files(temp_dirs, num_images=3)
        
        opts = BatchOptions(
            images_root=temp_dirs["images"],
            depths_root=temp_dirs["depths"],
            out_root=temp_dirs["output"],
            mode="haze",
            workers=1,
        )
        
        error_count = process_batch(opts)
        
        assert error_count == 0, "Expected no errors in successful batch"
        
        # Check output files were created
        output_files = list(Path(temp_dirs["output"]).glob("*_depthhaze.*"))
        assert len(output_files) == 3, "Expected 3 output files"
    
    def test_batch_with_missing_images(self, temp_dirs):
        """Test that batch handles missing source images gracefully"""
        # Create only depth maps, no source images
        for i in range(2):
            base = f"test_image_{i:03d}"
            depth_arr = np.random.rand(100, 100) * 65535
            depth = Image.fromarray(depth_arr.astype(np.uint16))
            depth_path = Path(temp_dirs["depths"]) / f"{base}_depth16.png"
            depth.save(depth_path)
        
        opts = BatchOptions(
            images_root=temp_dirs["images"],
            depths_root=temp_dirs["depths"],
            out_root=temp_dirs["output"],
            mode="clarity",
            workers=1,
            skip_missing=True,
        )
        
        error_count = process_batch(opts)
        
        # Should have errors due to missing source images
        assert error_count == 2, "Expected 2 errors for missing images"
        
        # No output files should be created
        output_files = list(Path(temp_dirs["output"]).glob("*"))
        assert len(output_files) == 0, "Expected no output files"
    
    def test_partial_failure_scenario(self, temp_dirs):
        """Test batch with some successes and some failures"""
        # Create 2 valid files
        create_test_files(temp_dirs, num_images=2)
        
        # Create a depth map without corresponding image
        depth_arr = np.random.rand(100, 100) * 65535
        depth = Image.fromarray(depth_arr.astype(np.uint16))
        depth_path = Path(temp_dirs["depths"]) / "missing_source_depth16.png"
        depth.save(depth_path)
        
        opts = BatchOptions(
            images_root=temp_dirs["images"],
            depths_root=temp_dirs["depths"],
            out_root=temp_dirs["output"],
            mode="dof",
            workers=1,
            skip_missing=True,
        )
        
        error_count = process_batch(opts)
        
        assert error_count == 1, "Expected 1 error for missing image"
        
        # Should have 2 successful outputs
        output_files = list(Path(temp_dirs["output"]).glob("*_depthdof.*"))
        assert len(output_files) == 2, "Expected 2 output files"


class TestExitCodes:
    """Test exit code behavior with different error scenarios"""
    
    def test_main_success_no_errors(self, temp_dirs):
        """Test main returns 0 when all files process successfully"""
        create_test_files(temp_dirs, num_images=2)
        
        argv = [
            "haze",
            temp_dirs["images"],
            temp_dirs["depths"],
            temp_dirs["output"],
        ]
        
        exit_code = main(argv)
        assert exit_code == 0, "Expected exit code 0 for successful batch"
    
    def test_main_strict_mode_with_errors(self, temp_dirs):
        """Test main returns 1 in strict mode (default) with errors"""
        # Create depth without corresponding image
        depth_arr = np.random.rand(100, 100) * 65535
        depth = Image.fromarray(depth_arr.astype(np.uint16))
        depth_path = Path(temp_dirs["depths"]) / "test_depth16.png"
        depth.save(depth_path)
        
        argv = [
            "clarity",
            temp_dirs["images"],
            temp_dirs["depths"],
            temp_dirs["output"],
        ]
        
        exit_code = main(argv)
        assert exit_code == 1, "Expected exit code 1 for errors in strict mode"
    
    def test_main_partial_success_mode_with_some_success(self, temp_dirs):
        """Test main returns 0 in partial success mode when at least one file succeeds"""
        # Create 2 valid files
        create_test_files(temp_dirs, num_images=2)
        
        # Create a depth map without corresponding image
        depth_arr = np.random.rand(100, 100) * 65535
        depth = Image.fromarray(depth_arr.astype(np.uint16))
        depth_path = Path(temp_dirs["depths"]) / "missing_depth16.png"
        depth.save(depth_path)
        
        argv = [
            "haze",
            temp_dirs["images"],
            temp_dirs["depths"],
            temp_dirs["output"],
            "--allow-partial-success",
        ]
        
        exit_code = main(argv)
        assert exit_code == 0, "Expected exit code 0 in partial success mode with some successes"
    
    def test_main_partial_success_mode_all_failures(self, temp_dirs):
        """Test main returns 1 in partial success mode when all files fail"""
        # Create only depth maps, no source images
        for i in range(2):
            depth_arr = np.random.rand(100, 100) * 65535
            depth = Image.fromarray(depth_arr.astype(np.uint16))
            depth_path = Path(temp_dirs["depths"]) / f"test_{i}_depth16.png"
            depth.save(depth_path)
        
        argv = [
            "dof",
            temp_dirs["images"],
            temp_dirs["depths"],
            temp_dirs["output"],
            "--allow-partial-success",
        ]
        
        exit_code = main(argv)
        assert exit_code == 1, "Expected exit code 1 even in partial success mode when all files fail"
    
    def test_main_no_depth_maps_found(self, temp_dirs):
        """Test main raises SystemExit when no depth maps are found"""
        argv = [
            "haze",
            temp_dirs["images"],
            temp_dirs["depths"],
            temp_dirs["output"],
        ]
        
        with pytest.raises(SystemExit):
            main(argv)


class TestBatchOptions:
    """Test BatchOptions dataclass"""
    
    def test_batch_options_defaults(self):
        """Test that BatchOptions has correct defaults"""
        opts = BatchOptions(
            images_root="/path/to/images",
            depths_root="/path/to/depths",
            out_root="/path/to/output",
        )
        
        assert opts.mode == "haze"
        assert opts.workers == 1
        assert opts.verbose is False
        assert opts.skip_missing is True
        assert opts.allow_partial_success is False
    
    def test_batch_options_allow_partial_success(self):
        """Test that allow_partial_success flag works"""
        opts = BatchOptions(
            images_root="/path/to/images",
            depths_root="/path/to/depths",
            out_root="/path/to/output",
            allow_partial_success=True,
        )
        
        assert opts.allow_partial_success is True


class TestCLIParsing:
    """Test CLI argument parsing"""
    
    def test_cli_help_includes_allow_partial_success(self, temp_dirs):
        """Test that --allow-partial-success flag appears in help"""
        argv = ["haze", "--help"]
        
        with pytest.raises(SystemExit) as exc_info:
            main(argv)
        
        # Help exits with code 0
        assert exc_info.value.code == 0
    
    def test_cli_allow_partial_success_flag(self, temp_dirs):
        """Test that --allow-partial-success flag is parsed correctly"""
        create_test_files(temp_dirs, num_images=1)
        
        argv = [
            "clarity",
            temp_dirs["images"],
            temp_dirs["depths"],
            temp_dirs["output"],
            "--allow-partial-success",
        ]
        
        exit_code = main(argv)
        assert exit_code == 0


class TestMultiprocessing:
    """Test multiprocessing functionality"""
    
    def test_batch_with_multiple_workers(self, temp_dirs):
        """Test that batch processing works with multiple workers"""
        create_test_files(temp_dirs, num_images=4)
        
        opts = BatchOptions(
            images_root=temp_dirs["images"],
            depths_root=temp_dirs["depths"],
            out_root=temp_dirs["output"],
            mode="haze",
            workers=2,  # Use 2 workers
        )
        
        error_count = process_batch(opts)
        
        assert error_count == 0, "Expected no errors with multiple workers"
        
        # Check all output files were created
        output_files = list(Path(temp_dirs["output"]).glob("*_depthhaze.*"))
        assert len(output_files) == 4, "Expected 4 output files"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
