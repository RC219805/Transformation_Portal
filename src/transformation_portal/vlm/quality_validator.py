"""Quality validation for enhanced architectural imagery.

Validates that AI enhancements maintain:
- Photographic realism
- Structural accuracy
- Material consistency
- Lighting plausibility
- Aesthetic quality

Uses LLaVA-1.5 for intelligent quality assessment.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from transformation_portal.vlm.llava import LLaVAProcessor

LLaVAProcessor: Any = None

logger = logging.getLogger(__name__)


def _resolve_llava_processor_class() -> Any:
    global LLaVAProcessor
    if LLaVAProcessor is None:
        from transformation_portal.vlm.llava import LLaVAProcessor as resolved_processor

        LLaVAProcessor = resolved_processor
    return LLaVAProcessor


class QualityAspect(Enum):
    """Aspects of image quality to validate."""

    REALISM = "realism"
    STRUCTURAL_ACCURACY = "structural_accuracy"
    MATERIAL_CONSISTENCY = "material_consistency"
    LIGHTING_PLAUSIBILITY = "lighting_plausibility"
    AESTHETIC_QUALITY = "aesthetic_quality"


class ValidationStatus(Enum):
    """Validation result status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class QualityScore:
    """Quality score for a specific aspect.

    Attributes:
        aspect: Quality aspect being scored
        score: Numerical score (0-10)
        status: Pass/warning/fail status
        comments: Specific feedback
        issues: List of identified issues
    """

    aspect: QualityAspect
    score: float
    status: ValidationStatus
    comments: str
    issues: List[str]


@dataclass
class ValidationReport:
    """Complete quality validation report.

    Attributes:
        overall_status: Overall validation status
        scores: Quality scores for each aspect
        overall_score: Average quality score (0-10)
        artifacts: List of detected artifacts
        recommendations: Improvement recommendations
        passed_validation: Whether image passed quality gates
        raw_assessment: Full LLaVA response
    """

    overall_status: ValidationStatus
    scores: List[QualityScore]
    overall_score: float
    artifacts: List[str]
    recommendations: List[str]
    passed_validation: bool
    raw_assessment: str


class QualityValidator:
    """Validate quality of enhanced architectural imagery.

    Uses LLaVA-1.5 to assess whether AI enhancements maintain
    photographic realism and architectural accuracy.

    Quality gates:
    - Realism score >= 7/10
    - No critical structural issues
    - Material consistency maintained
    - Lighting remains plausible
    - No obvious AI artifacts

    Example:
        >>> validator = QualityValidator()
        >>> report = validator.validate("enhanced_image.jpg")
        >>> if report.passed_validation:
        ...     print("Image passed quality validation")
        ... else:
        ...     print(f"Issues: {report.artifacts}")
    """

    # Quality thresholds
    PASS_THRESHOLD = 7.0  # Score >= 7 passes
    WARNING_THRESHOLD = 5.0  # Score 5-7 is warning
    # Score < 5 fails

    DETAILED_VALIDATION_PROMPT = """Perform a rigorous quality assessment of this architectural image.

Evaluate each aspect on a scale of 0-10 and identify any issues:

1. PHOTOGRAPHIC REALISM (0-10):
   - Does this look like a real photograph or does it have AI/CGI artifacts?
   - Are textures, reflections, and materials convincing?
   - Any uncanny valley or obviously synthetic elements?
   Score: __/10
   Issues: [list any]

2. STRUCTURAL ACCURACY (0-10):
   - Are walls, floors, ceilings geometrically correct?
   - Do architectural elements align properly?
   - Are proportions and perspective accurate?
   Score: __/10
   Issues: [list any]

3. MATERIAL CONSISTENCY (0-10):
   - Do materials look natural (proper texture, color, reflections)?
   - Is material rendering consistent across the image?
   - Any impossible or implausible material properties?
   Score: __/10
   Issues: [list any]

4. LIGHTING PLAUSIBILITY (0-10):
   - Is the lighting physically plausible?
   - Are shadows and highlights consistent?
   - Any unrealistic glows, halos, or lighting artifacts?
   Score: __/10
   Issues: [list any]

5. AESTHETIC QUALITY (0-10):
   - Overall visual appeal and composition
   - Color harmony and balance
   - Professional photography quality
   Score: __/10
   Issues: [list any]

ARTIFACTS: List any visible artifacts (halos, noise, distortions, inconsistencies, etc.)

RECOMMENDATIONS: Suggest specific improvements if needed.

OVERALL ASSESSMENT: Would this pass as professional architectural photography?

Provide detailed, honest assessment."""

    QUICK_VALIDATION_PROMPT = """Quickly assess this architectural image for quality:

1. Does it look photorealistic or are there obvious AI/CGI artifacts?
2. Are architectural structures geometrically accurate?
3. Do materials look natural and consistent?
4. Is the lighting plausible and well-balanced?
5. Overall aesthetic quality?

Rate overall: PASS / WARNING / FAIL
List any critical issues found."""

    def __init__(
        self,
        llava_processor: Optional["LLaVAProcessor"] = None,
        pass_threshold: float = 7.0,
        warning_threshold: float = 5.0,
        **llava_kwargs,
    ):
        """Initialize quality validator.

        Args:
            llava_processor: Existing LLaVA processor (creates new if None)
            pass_threshold: Minimum score to pass (default: 7.0)
            warning_threshold: Minimum score before failing (default: 5.0)
            **llava_kwargs: Arguments passed to LLaVAProcessor if creating new
        """
        if llava_processor is not None:
            self.processor = llava_processor
        else:
            self.processor = _resolve_llava_processor_class()(**llava_kwargs)

        self.pass_threshold = pass_threshold
        self.warning_threshold = warning_threshold

        logger.info(f"QualityValidator initialized " f"(pass>={pass_threshold}, warning>={warning_threshold})")

    def validate(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        detailed: bool = True,
        strict: bool = False,
    ) -> ValidationReport:
        """Validate image quality.

        Args:
            image: Image to validate
            detailed: Perform detailed assessment (slower but more thorough)
            strict: Use stricter validation criteria

        Returns:
            Complete validation report
        """
        # Get assessment from LLaVA
        prompt = self.DETAILED_VALIDATION_PROMPT if detailed else self.QUICK_VALIDATION_PROMPT

        raw_assessment = self.processor.analyze_image(
            image,
            prompt=prompt,
            temperature=0.1,  # Low temperature for consistent assessment
            max_new_tokens=1024 if detailed else 256,
        )

        # Parse assessment
        if detailed:
            scores = self._parse_detailed_scores(raw_assessment)
            artifacts = self._extract_artifacts(raw_assessment)
            recommendations = self._extract_recommendations(raw_assessment)
        else:
            scores = self._parse_quick_assessment(raw_assessment)
            artifacts = self._extract_issues_quick(raw_assessment)
            recommendations = []

        # Calculate overall score
        overall_score = np.mean([score.score for score in scores])

        # Determine overall status
        overall_status = self._determine_status(overall_score, scores, strict)

        # Check if passed validation
        passed = overall_status == ValidationStatus.PASS

        return ValidationReport(
            overall_status=overall_status,
            scores=scores,
            overall_score=overall_score,
            artifacts=artifacts,
            recommendations=recommendations,
            passed_validation=passed,
            raw_assessment=raw_assessment,
        )

    def validate_enhancement(
        self,
        original: Union[str, Path, Image.Image, np.ndarray],
        enhanced: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict[str, ValidationReport]:
        """Validate enhanced image against original.

        Checks that enhancement improved quality without introducing artifacts.

        Args:
            original: Original image
            enhanced: Enhanced image

        Returns:
            Dictionary with validation reports for both images and comparison
        """
        # Validate both images
        original_report = self.validate(original, detailed=True)
        enhanced_report = self.validate(enhanced, detailed=True)

        # Check for quality degradation
        quality_improved = enhanced_report.overall_score > original_report.overall_score
        new_artifacts = len(enhanced_report.artifacts) > len(original_report.artifacts)

        # Enhancement-specific validation
        enhancement_validation = {
            "quality_improved": quality_improved,
            "new_artifacts_introduced": new_artifacts,
            "score_delta": enhanced_report.overall_score - original_report.overall_score,
            "enhancement_valid": quality_improved and not new_artifacts,
        }

        return {
            "original": original_report,
            "enhanced": enhanced_report,
            "enhancement_validation": enhancement_validation,
        }

    def _parse_detailed_scores(self, text: str) -> List[QualityScore]:
        """Parse detailed assessment scores."""
        scores = []

        # Parse each aspect
        aspects = [
            (QualityAspect.REALISM, "photographic realism"),
            (QualityAspect.STRUCTURAL_ACCURACY, "structural accuracy"),
            (QualityAspect.MATERIAL_CONSISTENCY, "material consistency"),
            (QualityAspect.LIGHTING_PLAUSIBILITY, "lighting plausibility"),
            (QualityAspect.AESTHETIC_QUALITY, "aesthetic quality"),
        ]

        text_lower = text.lower()

        for aspect, aspect_name in aspects:
            # Extract score (looking for patterns like "Score: 8/10" or "8/10")
            score = self._extract_score(text_lower, aspect_name)

            # Extract issues
            issues = self._extract_issues_for_aspect(text, aspect_name)

            # Determine status
            status = self._score_to_status(score)

            # Extract comments for this aspect
            comments = self._extract_aspect_comments(text, aspect_name)

            scores.append(
                QualityScore(
                    aspect=aspect,
                    score=score,
                    status=status,
                    comments=comments,
                    issues=issues,
                )
            )

        return scores

    def _parse_quick_assessment(self, text: str) -> List[QualityScore]:
        """Parse quick assessment into scores."""
        # For quick assessment, estimate scores based on overall rating
        text_lower = text.lower()

        if "pass" in text_lower and "fail" not in text_lower:
            base_score = 8.0
            status = ValidationStatus.PASS
        elif "warning" in text_lower:
            base_score = 6.0
            status = ValidationStatus.WARNING
        elif "fail" in text_lower:
            base_score = 4.0
            status = ValidationStatus.FAIL
        else:
            base_score = 5.0
            status = ValidationStatus.UNKNOWN

        # Create scores for all aspects with base score
        aspects = [
            QualityAspect.REALISM,
            QualityAspect.STRUCTURAL_ACCURACY,
            QualityAspect.MATERIAL_CONSISTENCY,
            QualityAspect.LIGHTING_PLAUSIBILITY,
            QualityAspect.AESTHETIC_QUALITY,
        ]

        return [
            QualityScore(
                aspect=aspect,
                score=base_score,
                status=status,
                comments=text[:200],  # First 200 chars as comment
                issues=[],
            )
            for aspect in aspects
        ]

    def _extract_score(self, text: str, aspect_name: str) -> float:
        """Extract numerical score for an aspect."""
        import re

        # Look for score near aspect name
        aspect_section = self._extract_section(text, aspect_name, 500)

        # Pattern: Score: X/10 or X/10 or X.X/10
        score_patterns = [
            r"score:\s*(\d+(?:\.\d+)?)/10",
            r"(\d+(?:\.\d+)?)/10",
            r"score:\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, aspect_section)
            if match:
                try:
                    score = float(match.group(1))
                    return min(10.0, max(0.0, score))  # Clamp to 0-10
                except ValueError:
                    continue

        # Default to middle score if not found
        logger.warning(f"Could not extract score for {aspect_name}, defaulting to 5.0")
        return 5.0

    def _extract_section(self, text: str, section_name: str, length: int = 500) -> str:
        """Extract section of text after a section name."""
        text_lower = text.lower()
        section_name_lower = section_name.lower()

        if section_name_lower in text_lower:
            start = text_lower.index(section_name_lower)
            return text_lower[start : start + length]

        return text_lower[:length]

    def _extract_issues_for_aspect(self, text: str, aspect_name: str) -> List[str]:
        """Extract issues mentioned for a specific aspect."""
        # Look for "Issues:" after aspect name
        section = self._extract_section(text, aspect_name, 300)

        if "issues:" in section:
            issues_text = section.split("issues:")[-1].split("\n")[0]
            # Clean and split
            if "none" in issues_text.lower() or "n/a" in issues_text.lower():
                return []
            return [issues_text.strip()]

        return []

    def _extract_artifacts(self, text: str) -> List[str]:
        """Extract listed artifacts."""
        artifacts = []

        if "artifacts:" in text.lower():
            artifacts_section = text.lower().split("artifacts:")[-1].split("\n\n")[0]

            # Common artifact keywords
            artifact_keywords = [
                "halo",
                "haloing",
                "noise",
                "blur",
                "distortion",
                "artifact",
                "inconsistent",
                "unnatural",
                "synthetic",
                "fake",
                "unrealistic",
                "oversaturated",
                "overexposed",
                "underexposed",
                "banding",
                "compression",
                "posterization",
            ]

            for keyword in artifact_keywords:
                if keyword in artifacts_section:
                    artifacts.append(keyword)

        return artifacts

    def _extract_issues_quick(self, text: str) -> List[str]:
        """Extract issues from quick assessment."""
        issues = []

        text_lower = text.lower()

        # Look for "issues:" or "critical issues:"
        if "issues:" in text_lower or "critical" in text_lower:
            # Extract issues section
            for line in text.split("\n"):
                if any(word in line.lower() for word in ["issue", "problem", "artifact", "error"]):
                    issues.append(line.strip())

        return issues

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from assessment."""
        recommendations = []

        if "recommendations:" in text.lower():
            rec_section = text.split("RECOMMENDATIONS:")[-1].split("\n\n")[0]

            # Split into bullet points or lines
            for line in rec_section.split("\n"):
                line = line.strip()
                if line and len(line) > 10:  # Meaningful recommendation
                    # Remove bullet points
                    line = line.lstrip("- •*123456789.")
                    recommendations.append(line.strip())

        return recommendations

    def _extract_aspect_comments(self, text: str, aspect_name: str) -> str:
        """Extract comments for specific aspect."""
        section = self._extract_section(text, aspect_name, 400)

        # Get first few lines after aspect name
        lines = section.split("\n")[:3]
        comment = " ".join(lines).strip()

        return comment[:200]  # Limit length

    def _score_to_status(self, score: float) -> ValidationStatus:
        """Convert numerical score to validation status."""
        if score >= self.pass_threshold:
            return ValidationStatus.PASS
        elif score >= self.warning_threshold:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.FAIL

    def _determine_status(self, overall_score: float, scores: List[QualityScore], strict: bool) -> ValidationStatus:
        """Determine overall validation status."""
        # Check if any aspect failed critically
        failed_aspects = [s for s in scores if s.status == ValidationStatus.FAIL]

        if strict and failed_aspects:
            return ValidationStatus.FAIL

        if overall_score >= self.pass_threshold:
            return ValidationStatus.PASS
        elif overall_score >= self.warning_threshold:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.FAIL

    def create_validation_summary(self, report: ValidationReport) -> str:
        """Create human-readable validation summary.

        Args:
            report: Validation report

        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("Quality Validation Report")
        summary.append("=" * 50)
        summary.append(f"Overall Status: {report.overall_status.value.upper()}")
        summary.append(f"Overall Score: {report.overall_score:.1f}/10")
        summary.append(f"Passed: {'✓ YES' if report.passed_validation else '✗ NO'}")
        summary.append("")

        summary.append("Aspect Scores:")
        for score in report.scores:
            status_symbol = {
                ValidationStatus.PASS: "✓",
                ValidationStatus.WARNING: "⚠",
                ValidationStatus.FAIL: "✗",
                ValidationStatus.UNKNOWN: "?",
            }[score.status]

            summary.append(f"  {status_symbol} {score.aspect.value.replace('_', ' ').title()}: " f"{score.score:.1f}/10")

        if report.artifacts:
            summary.append("")
            summary.append("Artifacts Detected:")
            for artifact in report.artifacts:
                summary.append(f"  • {artifact}")

        if report.recommendations:
            summary.append("")
            summary.append("Recommendations:")
            for rec in report.recommendations:
                summary.append(f"  • {rec}")

        return "\n".join(summary)
