"""CPU/core parser tests for VLM quality validation."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import pytest

from transformation_portal.vlm.quality_validator import QualityValidator, ValidationStatus

pytestmark = [pytest.mark.unit]


class FakeProcessor:
    """Deterministic processor stub that never touches model runtimes."""

    def __init__(self, responses: str | Iterable[str]):
        if isinstance(responses, str):
            responses = [responses]
        self._responses = deque(responses)
        self.calls: list[dict[str, object]] = []

    def analyze_image(self, image, **kwargs):
        self.calls.append({"image": image, **kwargs})
        return self._responses.popleft()


def _validator(response: str | Iterable[str]) -> QualityValidator:
    return QualityValidator(llava_processor=FakeProcessor(response))


def _detailed_response(
    *,
    realism: str = "8/10",
    structural: str = "8/10",
    materials: str = "8/10",
    lighting: str = "8/10",
    aesthetic: str = "8/10",
    artifacts: str = "None",
    recommendations: str = "- Preserve current processing settings.",
) -> str:
    return f"""
1. PHOTOGRAPHIC REALISM (0-10):
   Score: {realism}
   Issues: None

2. STRUCTURAL ACCURACY (0-10):
   Score: {structural}
   Issues: None

3. MATERIAL CONSISTENCY (0-10):
   Score: {materials}
   Issues: None

4. LIGHTING PLAUSIBILITY (0-10):
   Score: {lighting}
   Issues: None

5. AESTHETIC QUALITY (0-10):
   Score: {aesthetic}
   Issues: None

ARTIFACTS: {artifacts}

RECOMMENDATIONS:
{recommendations}

OVERALL ASSESSMENT: Professional architectural photography.
"""


def test_processor_injection_uses_fake_processor_without_ml_runtime() -> None:
    fake = FakeProcessor(_detailed_response())
    validator = QualityValidator(llava_processor=fake)

    report = validator.validate("image-token", detailed=True)

    assert report.overall_status is ValidationStatus.PASS
    assert fake.calls[0]["image"] == "image-token"
    assert fake.calls[0]["temperature"] == 0.1
    assert fake.calls[0]["max_new_tokens"] == 1024


@pytest.mark.parametrize(
    ("text", "aspect", "expected"),
    [
        ("photographic realism\nScore: 8/10\nIssues: none", "photographic realism", 8.0),
        ("material consistency is acceptable at 8.5/10 today", "material consistency", 8.5),
        ("lighting plausibility has no numeric rating", "lighting plausibility", 5.0),
    ],
)
def test_extract_score_accepts_score_formats_and_defaults(
    text: str,
    aspect: str,
    expected: float,
) -> None:
    validator = _validator(_detailed_response())

    assert validator._extract_score(text.lower(), aspect) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (7.0, ValidationStatus.PASS),
        (5.0, ValidationStatus.WARNING),
        (4.99, ValidationStatus.FAIL),
    ],
)
def test_score_to_status_threshold_boundaries(score: float, expected: ValidationStatus) -> None:
    validator = _validator(_detailed_response())

    assert validator._score_to_status(score) is expected


def test_validate_strict_mode_fails_when_any_aspect_fails() -> None:
    validator = _validator(_detailed_response(realism="9/10", structural="4/10"))

    relaxed = validator.validate("image-token", detailed=True, strict=False)
    validator.processor = FakeProcessor(_detailed_response(realism="9/10", structural="4/10"))
    strict = validator.validate("image-token", detailed=True, strict=True)

    assert relaxed.overall_status is ValidationStatus.PASS
    assert strict.overall_status is ValidationStatus.FAIL
    assert strict.passed_validation is False


def test_extract_artifacts_covers_quality_keywords() -> None:
    validator = _validator(_detailed_response())
    text = """
ARTIFACTS: halo around windows, banding in the sky, blur on edges,
synthetic texture patches, and overexposed highlights.
"""

    artifacts = validator._extract_artifacts(text)

    assert {"halo", "banding", "blur", "synthetic", "overexposed"}.issubset(artifacts)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "RECOMMENDATIONS:\n- Reduce sharpening halos.\n- Preserve window detail.",
            ["Reduce sharpening halos.", "Preserve window detail."],
        ),
        (
            "recommendations:\n- lower exposure in the sky.\n* rebalance warm interior light.",
            ["lower exposure in the sky.", "rebalance warm interior light."],
        ),
        (
            "RECOMMENDATIONS:\n1. Reduce synthetic texture.\n2. Re-run material pass.",
            ["Reduce synthetic texture.", "Re-run material pass."],
        ),
    ],
)
def test_extract_recommendations_handles_headers_bullets_and_numbering(
    text: str,
    expected: list[str],
) -> None:
    validator = _validator(_detailed_response())

    assert validator._extract_recommendations(text) == expected


def test_validate_enhancement_detects_improvement_without_new_artifacts() -> None:
    validator = _validator(
        [
            _detailed_response(realism="6/10", structural="6/10", artifacts="minor blur"),
            _detailed_response(realism="8/10", structural="8/10", artifacts="minor blur"),
        ]
    )

    result = validator.validate_enhancement("original", "enhanced")

    assert result["enhancement_validation"] == {
        "quality_improved": True,
        "new_artifacts_introduced": False,
        "score_delta": pytest.approx(0.8),
        "enhancement_valid": True,
    }


def test_validate_enhancement_detects_regression_and_new_artifacts() -> None:
    validator = _validator(
        [
            _detailed_response(realism="8/10", structural="8/10", artifacts="None"),
            _detailed_response(realism="6/10", structural="6/10", artifacts="halo and banding"),
        ]
    )

    result = validator.validate_enhancement("original", "enhanced")

    assert result["enhancement_validation"]["quality_improved"] is False
    assert result["enhancement_validation"]["new_artifacts_introduced"] is True
    assert result["enhancement_validation"]["score_delta"] == pytest.approx(-0.8)
    assert result["enhancement_validation"]["enhancement_valid"] is False
