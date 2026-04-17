"""Tests for vlm/quality_validator.py module (Phase 5 coverage).

Tests for:
- QualityAspect enum
- ValidationStatus enum
- QualityScore dataclass
- ValidationReport dataclass
- QualityValidator class (mocked)

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Skip all tests if torch not available (required by LLaVA/VLM)
torch = pytest.importorskip("torch", reason="torch required for VLM tests")

pytestmark = [pytest.mark.unit, pytest.mark.ml]


class TestQualityAspect:
    """Test QualityAspect enum."""

    def test_all_aspects_defined(self):
        """Test all quality aspects are defined."""
        from transformation_portal.vlm.quality_validator import QualityAspect

        assert QualityAspect.REALISM.value == "realism"
        assert QualityAspect.STRUCTURAL_ACCURACY.value == "structural_accuracy"
        assert QualityAspect.MATERIAL_CONSISTENCY.value == "material_consistency"
        assert QualityAspect.LIGHTING_PLAUSIBILITY.value == "lighting_plausibility"
        assert QualityAspect.AESTHETIC_QUALITY.value == "aesthetic_quality"


class TestValidationStatus:
    """Test ValidationStatus enum."""

    def test_all_statuses_defined(self):
        """Test all validation statuses are defined."""
        from transformation_portal.vlm.quality_validator import ValidationStatus

        assert ValidationStatus.PASS.value == "pass"
        assert ValidationStatus.WARNING.value == "warning"
        assert ValidationStatus.FAIL.value == "fail"
        assert ValidationStatus.UNKNOWN.value == "unknown"


class TestQualityScore:
    """Test QualityScore dataclass."""

    def test_basic_creation(self):
        """Test basic quality score creation."""
        from transformation_portal.vlm.quality_validator import (
            QualityAspect,
            QualityScore,
            ValidationStatus,
        )

        score = QualityScore(
            aspect=QualityAspect.REALISM,
            score=8.5,
            status=ValidationStatus.PASS,
            comments="Good photorealistic quality",
            issues=[],
        )

        assert score.aspect == QualityAspect.REALISM
        assert score.score == 8.5
        assert score.status == ValidationStatus.PASS

    def test_score_with_issues(self):
        """Test quality score with issues."""
        from transformation_portal.vlm.quality_validator import (
            QualityAspect,
            QualityScore,
            ValidationStatus,
        )

        score = QualityScore(
            aspect=QualityAspect.LIGHTING_PLAUSIBILITY,
            score=4.0,
            status=ValidationStatus.FAIL,
            comments="Unrealistic lighting",
            issues=["Shadows inconsistent", "Reflections missing"],
        )

        assert len(score.issues) == 2


class TestValidationReport:
    """Test ValidationReport dataclass."""

    def test_basic_creation(self):
        """Test basic validation report creation."""
        from transformation_portal.vlm.quality_validator import (
            QualityAspect,
            QualityScore,
            ValidationReport,
            ValidationStatus,
        )

        scores = [
            QualityScore(
                aspect=QualityAspect.REALISM,
                score=8.0,
                status=ValidationStatus.PASS,
                comments="Good",
                issues=[],
            ),
        ]

        report = ValidationReport(
            overall_status=ValidationStatus.PASS,
            scores=scores,
            overall_score=8.0,
            artifacts=[],
            recommendations=[],
            passed_validation=True,
            raw_assessment="Raw VLM response",
        )

        assert report.overall_status == ValidationStatus.PASS
        assert report.passed_validation is True

    def test_report_with_artifacts(self):
        """Test report with artifacts detected."""
        from transformation_portal.vlm.quality_validator import (
            ValidationReport,
            ValidationStatus,
        )

        report = ValidationReport(
            overall_status=ValidationStatus.WARNING,
            scores=[],
            overall_score=6.0,
            artifacts=["halo", "noise"],
            recommendations=["Apply denoising"],
            passed_validation=False,
            raw_assessment="...",
        )

        assert len(report.artifacts) == 2
        assert "halo" in report.artifacts


class TestQualityValidatorMocked:
    """Test QualityValidator with mocked LLaVA processor."""

    @pytest.fixture
    def mock_llava_processor(self):
        """Create mocked LLaVA processor."""
        mock = MagicMock()
        mock.analyze_image = MagicMock(return_value="""
1. PHOTOGRAPHIC REALISM (0-10):
   Score: 8/10
   Issues: None

2. STRUCTURAL ACCURACY (0-10):
   Score: 9/10
   Issues: None

3. MATERIAL CONSISTENCY (0-10):
   Score: 7/10
   Issues: Minor texture inconsistency

4. LIGHTING PLAUSIBILITY (0-10):
   Score: 8/10
   Issues: None

5. AESTHETIC QUALITY (0-10):
   Score: 9/10
   Issues: None

ARTIFACTS: None detected

RECOMMENDATIONS: None needed

OVERALL ASSESSMENT: This image would pass as professional architectural photography.
""")
        return mock

    def test_validator_initialization(self, mock_llava_processor):
        """Test validator initialization."""
        from transformation_portal.vlm.quality_validator import QualityValidator

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)

        assert validator.pass_threshold == 7.0  # Default
        assert validator.warning_threshold == 5.0  # Default

    def test_validator_custom_thresholds(self, mock_llava_processor):
        """Test validator with custom thresholds."""
        from transformation_portal.vlm.quality_validator import QualityValidator

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(
                llava_processor=mock_llava_processor,
                pass_threshold=8.0,
                warning_threshold=6.0,
            )

        assert validator.pass_threshold == 8.0
        assert validator.warning_threshold == 6.0

    def test_validate_detailed(self, mock_llava_processor, tmp_path):
        """Test detailed validation."""
        from transformation_portal.vlm.quality_validator import (
            QualityValidator,
            ValidationStatus,
        )

        # Create test image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)
            report = validator.validate(img_path, detailed=True)

        assert hasattr(report, "overall_status")
        assert hasattr(report, "scores")
        assert hasattr(report, "overall_score")

    def test_validate_quick(self, tmp_path):
        """Test quick validation."""
        from transformation_portal.vlm.quality_validator import (
            QualityValidator,
            ValidationStatus,
        )

        # Create mock processor for quick validation
        mock = MagicMock()
        mock.analyze_image = MagicMock(return_value="PASS - Good quality image")

        # Create test image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img_path)

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock)
            report = validator.validate(img_path, detailed=False)

        assert report.overall_status in [
            ValidationStatus.PASS,
            ValidationStatus.WARNING,
            ValidationStatus.FAIL,
            ValidationStatus.UNKNOWN,
        ]

    def test_validate_enhancement(self, mock_llava_processor, tmp_path):
        """Test enhancement validation comparing original and enhanced."""
        from transformation_portal.vlm.quality_validator import QualityValidator

        # Create test images
        orig_path = tmp_path / "original.png"
        enh_path = tmp_path / "enhanced.png"
        Image.new("RGB", (100, 100), color="gray").save(orig_path)
        Image.new("RGB", (100, 100), color="white").save(enh_path)

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)
            result = validator.validate_enhancement(orig_path, enh_path)

        assert "original" in result
        assert "enhanced" in result
        assert "enhancement_validation" in result

    def test_score_to_status_pass(self, mock_llava_processor):
        """Test score to status conversion - pass."""
        from transformation_portal.vlm.quality_validator import (
            QualityValidator,
            ValidationStatus,
        )

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)
            status = validator._score_to_status(8.0)

        assert status == ValidationStatus.PASS

    def test_score_to_status_warning(self, mock_llava_processor):
        """Test score to status conversion - warning."""
        from transformation_portal.vlm.quality_validator import (
            QualityValidator,
            ValidationStatus,
        )

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)
            status = validator._score_to_status(6.0)

        assert status == ValidationStatus.WARNING

    def test_score_to_status_fail(self, mock_llava_processor):
        """Test score to status conversion - fail."""
        from transformation_portal.vlm.quality_validator import (
            QualityValidator,
            ValidationStatus,
        )

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)
            status = validator._score_to_status(3.0)

        assert status == ValidationStatus.FAIL

    def test_extract_score(self, mock_llava_processor):
        """Test score extraction from text."""
        from transformation_portal.vlm.quality_validator import QualityValidator

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)

            text = "photographic realism score: 8/10 - looks good"
            score = validator._extract_score(text.lower(), "photographic realism")

        assert score == 8.0

    def test_extract_score_no_match(self, mock_llava_processor):
        """Test score extraction returns default when not found."""
        from transformation_portal.vlm.quality_validator import QualityValidator

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)

            text = "some random text without scores"
            score = validator._extract_score(text, "realism")

        assert score == 5.0  # Default

    def test_extract_artifacts(self, mock_llava_processor):
        """Test artifact extraction."""
        from transformation_portal.vlm.quality_validator import QualityValidator

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)

            text = """
ARTIFACTS: Some halo effects visible, minor noise in shadows
"""
            artifacts = validator._extract_artifacts(text)

        assert "halo" in artifacts
        assert "noise" in artifacts

    def test_create_validation_summary(self, mock_llava_processor):
        """Test validation summary creation."""
        from transformation_portal.vlm.quality_validator import (
            QualityAspect,
            QualityScore,
            QualityValidator,
            ValidationReport,
            ValidationStatus,
        )

        with patch("transformation_portal.vlm.quality_validator.LLaVAProcessor"):
            validator = QualityValidator(llava_processor=mock_llava_processor)

            scores = [
                QualityScore(
                    aspect=QualityAspect.REALISM,
                    score=8.0,
                    status=ValidationStatus.PASS,
                    comments="Good",
                    issues=[],
                ),
            ]

            report = ValidationReport(
                overall_status=ValidationStatus.PASS,
                scores=scores,
                overall_score=8.0,
                artifacts=[],
                recommendations=[],
                passed_validation=True,
                raw_assessment="...",
            )

            summary = validator.create_validation_summary(report)

        assert "Quality Validation Report" in summary
        assert "PASS" in summary
        assert "8.0" in summary


class TestPromptConstants:
    """Test prompt constant definitions."""

    def test_detailed_prompt_exists(self):
        """Test detailed validation prompt exists."""
        from transformation_portal.vlm.quality_validator import QualityValidator

        assert hasattr(QualityValidator, "DETAILED_VALIDATION_PROMPT")
        prompt = QualityValidator.DETAILED_VALIDATION_PROMPT

        assert "PHOTOGRAPHIC REALISM" in prompt
        assert "STRUCTURAL ACCURACY" in prompt
        assert "MATERIAL CONSISTENCY" in prompt
        assert "LIGHTING PLAUSIBILITY" in prompt
        assert "AESTHETIC QUALITY" in prompt

    def test_quick_prompt_exists(self):
        """Test quick validation prompt exists."""
        from transformation_portal.vlm.quality_validator import QualityValidator

        assert hasattr(QualityValidator, "QUICK_VALIDATION_PROMPT")
        prompt = QualityValidator.QUICK_VALIDATION_PROMPT

        assert "PASS" in prompt or "pass" in prompt.lower()
        assert "FAIL" in prompt or "fail" in prompt.lower()
