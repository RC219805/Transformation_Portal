# V3 Orchestrator Hardening: Testing Strategy

This document defines the comprehensive testing strategy for validating the hardening fixes before production deployment.

---

## Test Pyramid

```
                    ┌─────────────────┐
                    │  Production     │  ← 1 final test
                    │  Validation     │
                    └─────────────────┘
                  ┌───────────────────────┐
                  │  Integration Tests    │  ← 10 tests
                  │  (End-to-End)         │
                  └───────────────────────┘
              ┌───────────────────────────────┐
              │  Component Tests              │  ← 30 tests
              │  (Module-level)               │
              └───────────────────────────────┘
          ┌─────────────────────────────────────┐
          │  Unit Tests                         │  ← 60 tests
          │  (Function-level)                   │
          └─────────────────────────────────────┘
```

**Total**: 101 tests across 4 levels

---

## Phase 1 Critical Tests (60 tests)

### 1.1 Path Sanitization Tests (15 tests)

**File**: `tests/test_path_sanitization.py`

```python
import pytest
from lux_depth_v3.enhance.orchestrator import (
    sanitize_path_component_nonlossy,
    make_output_key,
)

class TestNonLossySanitization:
    """Test non-lossy path component sanitization."""

    def test_alphanumeric_preserved(self):
        """Alphanumeric chars should pass through unchanged."""
        assert sanitize_path_component_nonlossy("kitchen123") == "kitchen123"

    def test_underscore_hyphen_preserved(self):
        """Underscores and hyphens should pass through."""
        assert sanitize_path_component_nonlossy("living-room_v2") == "living-room_v2"

    def test_single_dot_preserved(self):
        """Single dots should be preserved."""
        assert sanitize_path_component_nonlossy("room.1") == "room.1"

    def test_colon_encoded(self):
        """Colons should be percent-encoded."""
        assert sanitize_path_component_nonlossy("kitchen:1") == "kitchen%3A1"

    def test_slash_encoded(self):
        """Slashes should be percent-encoded."""
        assert sanitize_path_component_nonlossy("kitchen/1") == "kitchen%2F1"

    def test_backslash_encoded(self):
        """Backslashes should be percent-encoded."""
        assert sanitize_path_component_nonlossy("kitchen\\1") == "kitchen%5C1"

    def test_no_collision_special_chars(self):
        """Different special chars should produce different outputs."""
        colon = sanitize_path_component_nonlossy("kitchen:1")
        slash = sanitize_path_component_nonlossy("kitchen/1")
        backslash = sanitize_path_component_nonlossy("kitchen\\1")

        assert colon != slash
        assert slash != backslash
        assert colon != backslash

    def test_leading_dots_stripped(self):
        """Leading dots should be stripped (prevent hidden files)."""
        assert sanitize_path_component_nonlossy(".hidden") == "hidden"
        assert sanitize_path_component_nonlossy("...multiple") == "multiple"

    def test_double_dots_encoded(self):
        """Double dots should be encoded (prevent parent traversal)."""
        result = sanitize_path_component_nonlossy("parent..child")
        assert ".." not in result
        assert "%2E%2E" in result

    def test_empty_raises_error(self):
        """Empty component should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            sanitize_path_component_nonlossy("")

    def test_only_dots_raises_error(self):
        """Component with only dots should raise ValueError."""
        with pytest.raises(ValueError, match="empty after sanitization"):
            sanitize_path_component_nonlossy("...")

    def test_long_component_truncated(self):
        """Very long components should be truncated with hash suffix."""
        long_name = "a" * 250
        result = sanitize_path_component_nonlossy(long_name, max_length=200)

        assert len(result) <= 200
        assert "__" in result  # Hash suffix

    def test_unicode_encoded(self):
        """Unicode chars should be percent-encoded."""
        result = sanitize_path_component_nonlossy("café")
        assert "caf%C3%A9" in result or "café" not in result

    def test_deterministic(self):
        """Same input should always produce same output."""
        input_str = "kitchen:special/char\\test"
        result1 = sanitize_path_component_nonlossy(input_str)
        result2 = sanitize_path_component_nonlossy(input_str)
        assert result1 == result2

    def test_reversible_encoding(self):
        """Should be possible to decode back (for debugging)."""
        import urllib.parse

        input_str = "kitchen:1"
        encoded = sanitize_path_component_nonlossy(input_str)
        decoded = urllib.parse.unquote(encoded)
        assert decoded == input_str


class TestMakeOutputKey:
    """Test collision-free output key generation."""

    def test_flat_structure(self, tmp_path):
        """Flat directory should produce simple keys."""
        input_root = tmp_path / "renders"
        input_path = input_root / "kitchen.jpg"

        key = make_output_key(input_path, input_root)
        assert key == Path("kitchen")

    def test_nested_structure(self, tmp_path):
        """Nested directories should preserve structure."""
        input_root = tmp_path / "renders"
        input_path = input_root / "floor1" / "kitchen" / "view.jpg"

        key = make_output_key(input_path, input_root)
        assert key == Path("floor1/kitchen/view")

    def test_same_filename_different_dirs(self, tmp_path):
        """Same filename in different dirs should produce different keys."""
        input_root = tmp_path / "renders"

        path1 = input_root / "kitchen" / "view.jpg"
        path2 = input_root / "exterior" / "view.jpg"

        key1 = make_output_key(path1, input_root)
        key2 = make_output_key(path2, input_root)

        assert key1 != key2
        assert key1 == Path("kitchen/view")
        assert key2 == Path("exterior/view")

    def test_special_chars_in_path(self, tmp_path):
        """Special chars in directory names should be encoded."""
        input_root = tmp_path / "renders"
        input_path = input_root / "kitchen:special" / "view.jpg"

        key = make_output_key(input_path, input_root)
        assert "kitchen%3Aspecial" in str(key)

    def test_not_relative_to_root(self, tmp_path):
        """Path not relative to root should fall back to flat naming."""
        input_root = tmp_path / "renders"
        input_path = tmp_path / "other" / "kitchen.jpg"

        key = make_output_key(input_path, input_root)
        assert key == Path("kitchen")
```

---

### 1.2 Config Fingerprint Tests (12 tests)

**File**: `tests/test_config_fingerprint.py`

```python
from lux_depth_v3.enhance.manifest import ConfigFingerprint
from lux_depth_v3.enhance.config import EnhanceConfig, ModelVariant

class TestConfigFingerprint:
    """Test config fingerprinting for cache validation."""

    def test_same_config_same_hash(self):
        """Identical configs should produce same hash."""
        config1 = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            v2_preset="interior_luxury",
        )
        config2 = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            v2_preset="interior_luxury",
        )

        fp1 = ConfigFingerprint.from_config(config1).to_sha256()
        fp2 = ConfigFingerprint.from_config(config2).to_sha256()

        assert fp1 == fp2

    def test_different_model_different_hash(self):
        """Different model variant should change hash."""
        config1 = EnhanceConfig(model_variant=ModelVariant.METRIC_LARGE)
        config2 = EnhanceConfig(model_variant=ModelVariant.METRIC_SMALL)

        fp1 = ConfigFingerprint.from_config(config1).to_sha256()
        fp2 = ConfigFingerprint.from_config(config2).to_sha256()

        assert fp1 != fp2

    def test_different_v2_preset_different_hash(self):
        """Different V2 preset should change hash."""
        config1 = EnhanceConfig(v2_preset="interior_luxury")
        config2 = EnhanceConfig(v2_preset="production_ultra")

        fp1 = ConfigFingerprint.from_config(config1).to_sha256()
        fp2 = ConfigFingerprint.from_config(config2).to_sha256()

        assert fp1 != fp2

    def test_different_quantization_different_hash(self):
        """Different quantization method should change hash."""
        config1 = EnhanceConfig(depth_quantization="p1p99")
        config2 = EnhanceConfig(depth_quantization="minmax")

        fp1 = ConfigFingerprint.from_config(config1).to_sha256()
        fp2 = ConfigFingerprint.from_config(config2).to_sha256()

        assert fp1 != fp2

    def test_different_device_different_hash(self):
        """Different device should change hash."""
        config1 = EnhanceConfig(depth_device="cpu")
        config2 = EnhanceConfig(depth_device="cuda")

        fp1 = ConfigFingerprint.from_config(config1).to_sha256()
        fp2 = ConfigFingerprint.from_config(config2).to_sha256()

        assert fp1 != fp2

    def test_force_flags_ignored(self):
        """Force flags should NOT affect fingerprint."""
        config1 = EnhanceConfig(force_depth=True, force_v2=True)
        config2 = EnhanceConfig(force_depth=False, force_v2=False)

        fp1 = ConfigFingerprint.from_config(config1).to_sha256()
        fp2 = ConfigFingerprint.from_config(config2).to_sha256()

        assert fp1 == fp2  # Force flags don't affect output

    def test_timeout_ignored(self):
        """Timeout should NOT affect fingerprint."""
        config1 = EnhanceConfig(v2_timeout=300.0)
        config2 = EnhanceConfig(v2_timeout=600.0)

        fp1 = ConfigFingerprint.from_config(config1).to_sha256()
        fp2 = ConfigFingerprint.from_config(config2).to_sha256()

        assert fp1 == fp2  # Timeout doesn't affect output

    def test_deterministic(self):
        """Same config should always produce same hash."""
        config = EnhanceConfig(v2_preset="interior_luxury")

        hashes = [
            ConfigFingerprint.from_config(config).to_sha256()
            for _ in range(10)
        ]

        assert len(set(hashes)) == 1  # All identical

    def test_hash_is_sha256(self):
        """Hash should be valid SHA256 (64 hex chars)."""
        config = EnhanceConfig()
        fp = ConfigFingerprint.from_config(config).to_sha256()

        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_depth_config_subset(self):
        """Depth config fingerprint should only include depth params."""
        config1 = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            v2_preset="interior_luxury",
        )
        config2 = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            v2_preset="production_ultra",  # Different V2 preset
        )

        depth_fp1 = ConfigFingerprint.from_config(config1).depth_only().to_sha256()
        depth_fp2 = ConfigFingerprint.from_config(config2).depth_only().to_sha256()

        assert depth_fp1 == depth_fp2  # V2 preset doesn't affect depth fingerprint

    def test_v2_config_subset(self):
        """V2 config fingerprint should only include V2 params."""
        config1 = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            v2_preset="interior_luxury",
        )
        config2 = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,  # Different model
            v2_preset="interior_luxury",
        )

        v2_fp1 = ConfigFingerprint.from_config(config1).v2_only().to_sha256()
        v2_fp2 = ConfigFingerprint.from_config(config2).v2_only().to_sha256()

        assert v2_fp1 == v2_fp2  # Model variant doesn't affect V2 fingerprint

    def test_json_serializable(self):
        """Config fingerprint should be JSON serializable."""
        import json

        config = EnhanceConfig()
        fp = ConfigFingerprint.from_config(config)

        # Should not raise
        json_str = json.dumps(fp.to_dict())
        assert len(json_str) > 0
```

---

### 1.3 Resume Logic Tests (18 tests)

**File**: `tests/test_resume_logic.py`

```python
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from lux_depth_v3.enhance.orchestrator import EnhanceOrchestrator
from lux_depth_v3.input_manager import ImageInput

class TestDepthResume:
    """Test depth regeneration logic."""

    def test_missing_depth_regenerates(self, tmp_path, sample_config):
        """Missing depth file should trigger regeneration."""
        orch = EnhanceOrchestrator(sample_config, tmp_path)

        # Depth file doesn't exist
        should_skip = orch.should_skip_depth(
            depth_path=tmp_path / "depth.png",
            manifest_path=tmp_path / "manifest.json",
            image_input=ImageInput(tmp_path / "test.jpg"),
            current_config_fp="abc123",
        )

        assert should_skip == False

    def test_missing_manifest_regenerates(self, tmp_path, sample_config, create_depth_file):
        """Missing manifest should trigger regeneration."""
        orch = EnhanceOrchestrator(sample_config, tmp_path)

        depth_path = tmp_path / "depth.png"
        create_depth_file(depth_path)

        should_skip = orch.should_skip_depth(
            depth_path=depth_path,
            manifest_path=tmp_path / "manifest.json",  # Doesn't exist
            image_input=ImageInput(tmp_path / "test.jpg"),
            current_config_fp="abc123",
        )

        assert should_skip == False

    def test_input_hash_mismatch_regenerates(self, tmp_path, sample_config, create_test_image):
        """Changed input image should trigger regeneration."""
        orch = EnhanceOrchestrator(sample_config, tmp_path)

        input_path = tmp_path / "test.jpg"
        create_test_image(input_path)

        # First run
        result1 = orch.enhance_image(ImageInput(input_path))

        # Modify input (change one pixel)
        img = Image.open(input_path)
        pixels = img.load()
        pixels[0, 0] = (255, 0, 0)
        img.save(input_path)

        # Second run should regenerate
        with patch.object(orch.inference_engine, 'predict') as mock_predict:
            mock_predict.return_value = MagicMock(depth=np.random.rand(100, 100))
            orch.enhance_image(ImageInput(input_path))
            mock_predict.assert_called_once()

    def test_config_mismatch_regenerates(self, tmp_path, create_test_image):
        """Changed config should trigger regeneration."""
        from lux_depth_v3.enhance.config import EnhanceConfig

        input_path = tmp_path / "test.jpg"
        create_test_image(input_path)

        # First run with config A
        config_a = EnhanceConfig(depth_quantization="p1p99")
        orch_a = EnhanceOrchestrator(config_a, tmp_path)
        result1 = orch_a.enhance_image(ImageInput(input_path))

        # Second run with config B
        config_b = EnhanceConfig(depth_quantization="minmax")
        orch_b = EnhanceOrchestrator(config_b, tmp_path)

        with patch.object(orch_b.inference_engine, 'predict') as mock_predict:
            mock_predict.return_value = MagicMock(depth=np.random.rand(100, 100))
            orch_b.enhance_image(ImageInput(input_path))
            mock_predict.assert_called_once()

    def test_corrupt_depth_regenerates(self, tmp_path, sample_config, create_test_image):
        """Corrupt depth file should trigger regeneration."""
        orch = EnhanceOrchestrator(sample_config, tmp_path)

        input_path = tmp_path / "test.jpg"
        create_test_image(input_path)

        # Create corrupt depth file (wrong dtype)
        depth_path = tmp_path / "depth" / "test_depth.png"
        depth_path.parent.mkdir(parents=True)

        corrupt_depth = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        Image.fromarray(corrupt_depth).save(depth_path)

        # Should regenerate
        with patch.object(orch.inference_engine, 'predict') as mock_predict:
            mock_predict.return_value = MagicMock(depth=np.random.rand(100, 100))
            orch.enhance_image(ImageInput(input_path))
            mock_predict.assert_called_once()

    def test_valid_depth_skipped(self, tmp_path, sample_config, create_test_image):
        """Valid depth with matching config should be skipped."""
        orch = EnhanceOrchestrator(sample_config, tmp_path)

        input_path = tmp_path / "test.jpg"
        create_test_image(input_path)

        # First run
        result1 = orch.enhance_image(ImageInput(input_path))

        # Second run (no changes)
        with patch.object(orch.inference_engine, 'predict') as mock_predict:
            orch.enhance_image(ImageInput(input_path))
            mock_predict.assert_not_called()  # Skipped ✅


class TestV2Resume:
    """Test V2 regeneration logic."""

    def test_depth_changed_forces_v2_rerun(self, tmp_path, create_test_image):
        """If depth regenerated, V2 must rerun."""
        from lux_depth_v3.enhance.config import EnhanceConfig

        input_path = tmp_path / "test.jpg"
        create_test_image(input_path)

        # First run
        config_a = EnhanceConfig(depth_quantization="p1p99")
        orch_a = EnhanceOrchestrator(config_a, tmp_path)
        result1 = orch_a.enhance_image(ImageInput(input_path))

        # Second run: change depth config (forces depth regen)
        config_b = EnhanceConfig(depth_quantization="minmax")
        orch_b = EnhanceOrchestrator(config_b, tmp_path)

        with patch.object(orch_b.v2_runner, 'run') as mock_v2:
            mock_v2.return_value = {"status": "ok", "runtime_s": 10.0}
            orch_b.enhance_image(ImageInput(input_path))
            mock_v2.assert_called_once()  # V2 must rerun

    def test_v2_config_changed_reruns_v2_only(self, tmp_path, create_test_image):
        """V2 config change should rerun V2 but skip depth."""
        from lux_depth_v3.enhance.config import EnhanceConfig

        input_path = tmp_path / "test.jpg"
        create_test_image(input_path)

        # First run
        config_a = EnhanceConfig(v2_preset="interior_luxury")
        orch_a = EnhanceOrchestrator(config_a, tmp_path)
        result1 = orch_a.enhance_image(ImageInput(input_path))

        # Second run: change V2 config only
        config_b = EnhanceConfig(v2_preset="production_ultra")
        orch_b = EnhanceOrchestrator(config_b, tmp_path)

        with patch.object(orch_b.inference_engine, 'predict') as mock_depth:
            with patch.object(orch_b.v2_runner, 'run') as mock_v2:
                mock_v2.return_value = {"status": "ok", "runtime_s": 10.0}
                orch_b.enhance_image(ImageInput(input_path))

                mock_depth.assert_not_called()  # Depth skipped ✅
                mock_v2.assert_called_once()     # V2 rerun ✅

    # ... 10 more V2 resume tests ...
```

---

### 1.4 Atomic Write Tests (9 tests)

**File**: `tests/test_atomic_writes.py`

```python
import pytest
from unittest.mock import patch
import numpy as np
from pathlib import Path

from lux_depth_v3.enhance.depth_writer import atomic_write_depth_u16_png
from lux_depth_v3.enhance.manifest import atomic_write_json

class TestAtomicDepthWrites:
    """Test atomic depth file writes."""

    def test_successful_write(self, tmp_path):
        """Normal write should succeed."""
        output_path = tmp_path / "depth.png"
        depth = np.random.rand(100, 100).astype(np.float32)

        p1, p99 = atomic_write_depth_u16_png(output_path, depth)

        assert output_path.exists()
        assert not (tmp_path / "depth.tmp.png").exists()  # Temp cleaned up

    def test_crash_cleanup(self, tmp_path):
        """Crash during write should clean up temp file."""
        output_path = tmp_path / "depth.png"
        depth = np.random.rand(100, 100).astype(np.float32)

        with patch('PIL.Image.Image.save', side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                atomic_write_depth_u16_png(output_path, depth)

        # No files should remain
        assert not output_path.exists()
        assert not (tmp_path / "depth.tmp.png").exists()

    def test_preserves_existing_on_crash(self, tmp_path):
        """Failed write should not corrupt existing file."""
        output_path = tmp_path / "depth.png"

        # Write initial valid file
        depth1 = np.random.rand(100, 100).astype(np.float32)
        p1, p99 = atomic_write_depth_u16_png(output_path, depth1)

        original_size = output_path.stat().st_size
        original_mtime = output_path.stat().st_mtime

        # Attempt to overwrite with crashing write
        import time
        time.sleep(0.1)  # Ensure mtime would change

        with patch('PIL.Image.Image.save', side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                atomic_write_depth_u16_png(output_path, depth1 * 2)

        # Original file should be unchanged
        assert output_path.exists()
        assert output_path.stat().st_size == original_size
        assert output_path.stat().st_mtime == original_mtime

    def test_parent_dir_created(self, tmp_path):
        """Parent directories should be created automatically."""
        output_path = tmp_path / "nested" / "deep" / "depth.png"
        depth = np.random.rand(100, 100).astype(np.float32)

        atomic_write_depth_u16_png(output_path, depth)

        assert output_path.exists()
        assert output_path.parent.exists()

    # ... 5 more atomic write tests ...


class TestAtomicJSONWrites:
    """Test atomic JSON manifest writes."""

    def test_successful_json_write(self, tmp_path):
        """Normal JSON write should succeed."""
        output_path = tmp_path / "manifest.json"
        data = {"key": "value", "number": 42}

        atomic_write_json(output_path, data)

        assert output_path.exists()
        assert not (tmp_path / "manifest.tmp.json").exists()

    # ... 3 more JSON write tests ...
```

---

### 1.5 EXIF Orientation Tests (6 tests)

**File**: `tests/test_exif_orientation.py`

```python
import pytest
from PIL import Image
import cv2
import numpy as np

from lux_depth_v3.enhance.preprocessing import normalize_exif_orientation

class TestEXIFNormalization:
    """Test EXIF orientation pre-normalization."""

    def test_orientation_6_rotated(self, tmp_path):
        """Orientation 6 (90° CW) should rotate dimensions."""
        # Create 100x200 image with orientation 6
        img = Image.new("RGB", (100, 200), color="red")
        exif = img.getexif()
        exif[0x0112] = 6  # Rotate 90° CW

        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.png"
        img.save(input_path, exif=exif)

        was_normalized = normalize_exif_orientation(input_path, output_path)

        assert was_normalized == True

        # Verify dimensions rotated
        img_norm = Image.open(output_path)
        assert img_norm.size == (200, 100)  # Width/height swapped

    def test_orientation_tag_removed(self, tmp_path):
        """EXIF orientation tag should be removed after normalization."""
        img = Image.new("RGB", (100, 200))
        exif = img.getexif()
        exif[0x0112] = 6

        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.png"
        img.save(input_path, exif=exif)

        normalize_exif_orientation(input_path, output_path)

        img_norm = Image.open(output_path)
        exif_norm = img_norm.getexif()
        assert 0x0112 not in exif_norm

    def test_pil_opencv_consistency(self, tmp_path):
        """PIL and OpenCV should see same dimensions after normalization."""
        # Create image with orientation
        img = Image.new("RGB", (100, 200), color=(255, 0, 0))
        exif = img.getexif()
        exif[0x0112] = 6

        input_path = tmp_path / "input.jpg"
        normalized_path = tmp_path / "normalized.png"
        img.save(input_path, exif=exif)

        normalize_exif_orientation(input_path, normalized_path)

        # Read with PIL (DA3 simulation)
        from PIL import ImageOps
        img_pil = Image.open(normalized_path)
        img_pil = ImageOps.exif_transpose(img_pil)  # Should be no-op

        # Read with OpenCV (V2 simulation)
        img_cv = cv2.imread(str(normalized_path))

        # Compare dimensions
        assert img_pil.size[0] == img_cv.shape[1]  # Width
        assert img_pil.size[1] == img_cv.shape[0]  # Height

    # ... 3 more EXIF tests ...
```

---

## Integration Tests (10 tests)

**File**: `tests/test_orchestrator_integration.py`

```python
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_single_image(self, tmp_path, create_test_image):
        """Process single image through full V3+V2 pipeline."""
        input_path = tmp_path / "input.jpg"
        create_test_image(input_path)

        config = EnhanceConfig()
        orch = EnhanceOrchestrator(config, tmp_path / "output")

        result = orch.enhance_image(ImageInput(input_path))

        assert result["status"] == "ok"
        assert Path(result["depth_path"]).exists()
        assert Path(result["manifest"]).exists()

    def test_full_pipeline_nested_batch(self, tmp_path, create_test_images):
        """Process nested directory structure."""
        # Create structure:
        #   input/
        #     kitchen/view.jpg
        #     exterior/view.jpg
        #     floor2/bedroom/view.jpg

        input_dir = tmp_path / "input"
        create_test_images([
            input_dir / "kitchen" / "view.jpg",
            input_dir / "exterior" / "view.jpg",
            input_dir / "floor2" / "bedroom" / "view.jpg",
        ])

        config = EnhanceConfig()
        orch = EnhanceOrchestrator(config, tmp_path / "output")

        results = orch.enhance_batch(input_dir)

        assert len(results) == 3
        assert all(r["status"] == "ok" for r in results)

        # Verify no collisions
        output_dir = tmp_path / "output" / "depth"
        assert (output_dir / "kitchen" / "view_depth.png").exists()
        assert (output_dir / "exterior" / "view_depth.png").exists()
        assert (output_dir / "floor2" / "bedroom" / "view_depth.png").exists()

    def test_resume_after_crash(self, tmp_path, create_test_images):
        """Crash and resume should work correctly."""
        input_dir = tmp_path / "input"
        images = create_test_images([
            input_dir / "img1.jpg",
            input_dir / "img2.jpg",
            input_dir / "img3.jpg",
        ])

        config = EnhanceConfig()
        orch = EnhanceOrchestrator(config, tmp_path / "output")

        # Process first two images
        orch.enhance_image(ImageInput(images[0]))
        orch.enhance_image(ImageInput(images[1]))

        # Simulate crash (don't process img3)

        # Resume: process all three
        results = orch.enhance_batch(input_dir)

        # First two should be skipped (resume)
        # Third should be processed
        # (Would need to mock and verify, simplified here)
        assert len(results) == 3

    # ... 7 more integration tests ...
```

---

## Production Validation (1 test)

**File**: `tests/test_production_validation.py`

```python
import pytest

@pytest.mark.slow
@pytest.mark.production
class TestProductionValidation:
    """Final production validation test."""

    def test_100_image_batch(self, tmp_path, download_test_dataset):
        """Process 100 diverse real-world images."""
        # Download test dataset (100 images, various sizes/formats)
        input_dir = download_test_dataset(tmp_path / "dataset")

        config = EnhanceConfig()
        orch = EnhanceOrchestrator(config, tmp_path / "output")

        results = orch.enhance_batch(input_dir)

        # Validate results
        assert len(results) == 100
        success_rate = sum(1 for r in results if r["status"] == "ok") / 100
        assert success_rate >= 0.95  # 95% success rate

        # Verify no collisions
        depth_files = list((tmp_path / "output" / "depth").rglob("*_depth.png"))
        assert len(depth_files) == len([r for r in results if r["status"] == "ok"])

        # Verify no temp files left behind
        tmp_files = list((tmp_path / "output").rglob("*.tmp.*"))
        assert len(tmp_files) == 0

        # Verify batch manifest
        batch_manifests = list((tmp_path / "output" / "manifests").glob("batch_*.json"))
        assert len(batch_manifests) >= 1
```

---

## CI Configuration

**File**: `.github/workflows/test-hardening.yml`

```yaml
name: V3 Orchestrator Hardening Tests

on:
  push:
    branches: [main, develop]
    paths:
      - 'lux_depth_v3/enhance/**'
      - 'tests/test_orchestrator_*.py'
  pull_request:
    branches: [main, develop]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e lux_depth_v3[test]

      - name: Run Phase 1 unit tests
        run: |
          pytest tests/test_path_sanitization.py -v --cov=lux_depth_v3.enhance
          pytest tests/test_config_fingerprint.py -v --cov=lux_depth_v3.enhance --cov-append
          pytest tests/test_resume_logic.py -v --cov=lux_depth_v3.enhance --cov-append
          pytest tests/test_atomic_writes.py -v --cov=lux_depth_v3.enhance --cov-append
          pytest tests/test_exif_orientation.py -v --cov=lux_depth_v3.enhance --cov-append

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -e lux_depth_v3[test]
          pip install -e lux_depth_v2[test]

      - name: Run integration tests
        run: |
          pytest tests/test_orchestrator_integration.py -v --maxfail=1

  production-validation:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'pull_request' && contains(github.event.pull_request.labels.*.name, 'pre-production')

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -e lux_depth_v3[test]
          pip install -e lux_depth_v2[test]

      - name: Run production validation
        run: |
          pytest tests/test_production_validation.py -v -m production
```

---

## Coverage Requirements

**Minimum coverage thresholds:**

| Module | Coverage | Critical Lines |
|--------|----------|---------------|
| `orchestrator.py` | 95% | All path generation, resume logic |
| `manifest.py` | 90% | Config fingerprint, atomic writes |
| `depth_writer.py` | 95% | Atomic write, verification |
| `security.py` | 100% | All sanitization functions |
| `preprocessing.py` | 90% | EXIF normalization |

**Enforcement:**

```bash
# Fail CI if coverage below threshold
pytest --cov=lux_depth_v3.enhance --cov-fail-under=90
```

---

## Manual Testing Checklist

Before production deployment, manually verify:

- [ ] Process 100+ diverse images (JPEG, PNG, TIFF)
- [ ] Nested directories 5+ levels deep
- [ ] Filenames with special chars (`:`, `/`, unicode)
- [ ] EXIF orientation variants (1-8)
- [ ] Large files (50MP+)
- [ ] Crash recovery (kill process mid-batch)
- [ ] Resume with config changes
- [ ] Concurrent batches (different dirs)
- [ ] Disk space exhaustion handling
- [ ] Verify no `.tmp.*` files left behind
- [ ] Review batch manifests for anomalies
- [ ] Performance regression test (vs. baseline)

---

## Success Criteria

Phase 1 hardening is **production-ready** when:

1. ✅ **All 60 unit tests pass** (100% success rate)
2. ✅ **All 10 integration tests pass** (100% success rate)
3. ✅ **Production validation passes** (≥95% success rate on 100 images)
4. ✅ **Coverage ≥90%** on critical modules
5. ✅ **No regressions** in existing tests
6. ✅ **Manual checklist 100% complete**
7. ✅ **CI green** on all Python versions (3.10, 3.11, 3.12)

**Deployment gate**: All 7 criteria must be met before production deployment.
