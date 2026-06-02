"""Tests to verify the repository restructuring is working correctly.

Note: These tests assume the package is installed in development mode.
Run `pip install -e .` from the repository root before running tests.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_depth_module_import():
    """Test that depth module can be imported from new location."""
    try:
        from transformation_portal.depth import __version__

        assert __version__ is not None
    except ImportError as e:
        # Allow missing dependencies for depth module
        if "tqdm" not in str(e) and "torch" not in str(e) and "PIL" not in str(e):
            raise


def test_material_response_import():
    """Test that material_response can be imported from new location."""
    from transformation_portal.processors.material_response.core import _clamp

    # Test basic functionality
    assert _clamp(5, 0, 10) == 5
    assert _clamp(-5, 0, 10) == 0
    assert _clamp(15, 0, 10) == 10


def test_asset_paths_exist():
    """Test that asset directories exist in new locations."""
    repo_root = Path(__file__).parent.parent

    # Check LUT directories
    assert (repo_root / "assets" / "luts" / "film_emulation").is_dir()
    assert (repo_root / "assets" / "luts" / "location_aesthetic").is_dir()
    assert (repo_root / "assets" / "luts" / "material_response").is_dir()

    # Check brand assets
    assert (repo_root / "assets" / "brand" / "lantern_logo").is_dir()

    # Check material texture assets
    assert (repo_root / "assets" / "textures" / "board_materials").is_dir()
    assert not (repo_root / "textures").exists()

    # Check projects
    assert (repo_root / "assets" / "projects").is_dir()


def test_documentation_consolidated():
    """Test that documentation is consolidated in docs/."""
    repo_root = Path(__file__).parent.parent

    # Check that docs/ exists
    assert (repo_root / "docs").is_dir()

    # Check specific documentation directories
    assert (repo_root / "docs" / "version_history").is_dir()
    assert (repo_root / "docs" / "brand").is_dir()

    # Check that old 08_Documentation doesn't exist
    assert not (repo_root / "08_Documentation").exists()


def test_scripts_directory():
    """Test that utility scripts are in scripts/."""
    repo_root = Path(__file__).parent.parent

    # Check scripts directory exists
    assert (repo_root / "scripts").is_dir()

    # Check that specific scripts are there
    assert (repo_root / "scripts" / "codebase_philosophy_auditor.py").is_file()
    assert (repo_root / "scripts" / "decision_decay_dashboard.py").is_file()


def test_old_numbered_directories_removed():
    """Test that old numbered directories are removed."""
    repo_root = Path(__file__).parent.parent

    # These should not exist anymore
    assert not (repo_root / "01_Film_Emulation").exists()
    assert not (repo_root / "02_Location_Aesthetic").exists()
    assert not (repo_root / "03_Material_Response").exists()
    assert not (repo_root / "09_Client_Deliverables").exists()


def test_depth_pipeline_moved():
    """Test that depth_pipeline is in src/transformation_portal/depth/."""
    repo_root = Path(__file__).parent.parent

    # Old location should not exist
    assert not (repo_root / "depth_pipeline").exists()

    # New location should exist
    assert (repo_root / "src" / "transformation_portal" / "depth").is_dir()
    assert (repo_root / "src" / "transformation_portal" / "depth" / "pipeline.py").is_file()
    assert (repo_root / "src" / "transformation_portal" / "depth" / "tools.py").is_file()


def test_specific_lut_files_exist():
    """Test that specific LUT files exist in new locations."""
    repo_root = Path(__file__).parent.parent

    # Test some key LUT files
    kodak_lut = repo_root / "assets" / "luts" / "film_emulation" / "Kodak" / "Kodak_2393_D55.cube"
    assert kodak_lut.is_file(), f"Expected LUT file not found: {kodak_lut}"

    montecito_lut = repo_root / "assets" / "luts" / "location_aesthetic" / "California" / "Montecito_Golden_Hour_HDR.cube"
    assert montecito_lut.is_file(), f"Expected LUT file not found: {montecito_lut}"


def test_utils_image_utils_exists():
    """Test that image_utils is properly located in utils."""
    repo_root = Path(__file__).parent.parent

    # Check that image_utils exists in the package
    image_utils = repo_root / "src" / "transformation_portal" / "utils" / "image_utils.py"
    assert image_utils.is_file(), f"Expected image_utils not found: {image_utils}"

    # Verify it contains the expected functions
    content = image_utils.read_text()
    assert "def load_image(" in content
    assert "def save_image(" in content
    assert "def pil_to_np(" in content
    assert "def np_to_pil(" in content
    assert "def load_image_rgb(" in content
