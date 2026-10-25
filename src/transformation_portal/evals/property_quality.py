"""Registered property-image quality evaluation, at native resolution.

Run ``python -m transformation_portal.evals.property_quality --help``. An
external JSON evalset registers exact reference/candidate file hashes, shared
working encoding, lighting group, scene tags and native crop coordinates. Real
references require a hash-bound review state and a rights record. Explicit
pending references can produce advisory diagnostics, but never reviewed coverage.
No private images or generated reports belong in Git.

Evalset shape (all paths relative to the external asset root)::

    {"schema": "tp.property.quality_evalset.v1", "evalset_id": "property-a",
     "fixtures": [{"fixture_id": "kitchen", "reference": {"path": "ref.tif",
       "sha256": "<64 lowercase hex>"}, "candidate": {"path": "final.tif",
       "sha256": "<64 lowercase hex>"}, "scene_tags": ["interior", "mixed_light"],
       "lighting_group": "daylight", "working_color_space": "sRGB",
       "working_transfer_function": "srgb", "synthetic_data": false,
       "rights_record": "operator-held permission record",
       "reference_review": {"decision": "approved_reference", "actor": "reviewer",
         "reviewed_at": "2026-10-24T20:00:00Z", "sha256": "<reference hash>"},
       "crops": [{"crop_id": "window", "box": [0, 0, 64, 64],
         "concerns": ["window_edges", "halos"]}]}]}

Boxes are [left, top, right, bottom], right/bottom exclusive, in the decoded
master's native pixel grid. No EXIF rotation, resizing, color conversion, model
inference or automatic approval occurs. Registered pairs must already have the
same orientation, dimensions and working encoding. Coverage records expose gaps;
even complete coverage is not representative photographic production acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from transformation_portal.evals.apex_metrics import ALGORITHM_REVISION, compute_apex_metrics
from transformation_portal.ingest.canonical_json import dump_json

EVALSET_SCHEMA = "tp.property.quality_evalset.v1"
REPORT_SCHEMA = "tp.property.quality_report.v1"
REQUIRED_SCENES = frozenset({"interior", "exterior", "mixed_light", "twilight"})
REQUIRED_CONCERNS = frozenset({"sky", "ceiling_cove", "fixtures", "window_edges", "railings", "foliage", "plaster"})
MAX_IMAGE_PIXELS = 40_000_000
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError(f"{field} must be a nonempty string of at most 1024 characters")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError("artifact sha256 must be lowercase 64-character hexadecimal")
    return value


def _labels(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 64:
        raise ValueError(f"{field} must contain 1..64 labels")
    labels = [_text(label, field) for label in value]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{field} contains duplicate labels")
    return labels


def _asset_path(root: Path, value: Any) -> Path:
    text = _text(value, "asset path")
    path = PurePosixPath(text)
    if path.is_absolute() or "\\" in text or any(part in {"", ".", ".."} for part in text.split("/")):
        raise ValueError("asset path must be a safe relative POSIX path")
    candidate = root.joinpath(*path.parts)
    if any(parent.is_symlink() for parent in [candidate, *candidate.parents] if parent != root.parent):
        raise ValueError("asset paths cannot contain symlinks")
    if not candidate.resolve(strict=True).is_relative_to(root):
        raise ValueError("asset path escapes the external root")
    return candidate


def _file_digest(handle: Any) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _check_geometry(width: int, height: int) -> None:
    if min(width, height) < 1 or max(width, height) > 65536 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("property evaluation exceeds the bounded native image geometry")


def _load_image(root: Path, spec: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(spec, Mapping) or set(spec) != {"path", "sha256"}:
        raise ValueError("artifact needs exactly path and sha256")
    path, expected = _asset_path(root, spec["path"]), _digest(spec["sha256"])
    icc_profile = None
    with path.open("rb") as handle:
        if _file_digest(handle) != expected:
            raise ValueError(f"artifact checksum mismatch: {spec['path']}")
        if path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile

            with tifffile.TiffFile(handle) as tiff:
                if len(tiff.pages) != 1:
                    raise ValueError("property evaluation requires single-frame TIFF masters")
                _check_geometry(tiff.pages[0].imagewidth, tiff.pages[0].imagelength)
                orientation = tiff.pages[0].tags.get("Orientation")
                if orientation is not None and orientation.value != 1:
                    raise ValueError("normalize image orientation before registering the pair")
                profile_tag = tiff.pages[0].tags.get(34675)
                icc_profile = profile_tag.value if profile_tag else None
                array = np.asarray(tiff.asarray())
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            from PIL import Image

            if path.suffix.lower() == ".png":
                header = handle.read(25)
                handle.seek(0)
                if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) == 25 and header[24] > 8:
                    raise ValueError("use a TIFF master for precision-preserving >8-bit RGB evaluation")
            with Image.open(handle) as image:
                _check_geometry(*image.size)
                if image.mode not in {"RGB", "RGBA", "L", "I;16", "I;16B", "I;16L"}:
                    raise ValueError("image must already use RGB or grayscale working encoding")
                if image.getexif().get(274, 1) != 1:
                    raise ValueError("normalize image orientation before registering the pair")
                icc_profile = image.info.get("icc_profile")
                array = np.asarray(image).copy()
        else:
            raise ValueError("quality references support TIFF, PNG and JPEG")
        if _file_digest(handle) != expected:
            raise ValueError("artifact changed during evaluation load")
    if array.ndim != 3 or array.shape[2] != 3 or not all(array.shape):
        raise ValueError("property evaluation requires single-frame RGB masters")
    if array.dtype not in (np.uint8, np.uint16, np.float32, np.float64) or not np.all(np.isfinite(array)):
        raise ValueError("unsupported or non-finite image samples")
    if array.dtype.kind == "f" and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError("float working images must use the declared normalized 0..1 contract")
    return array, {
        "path": spec["path"],
        "sha256": expected,
        "dtype": str(array.dtype),
        "dimensions": list(array.shape[1::-1]),
        "icc_profile_sha256": hashlib.sha256(icc_profile).hexdigest() if icc_profile else None,
    }


def _validate_reference_review(fixture: Mapping[str, Any], reference_sha: str) -> str:
    if not isinstance(fixture.get("synthetic_data"), bool):
        raise ValueError("synthetic_data must be explicit true or false")
    if fixture["synthetic_data"]:
        return "synthetic_protocol_only"
    _text(fixture.get("rights_record"), "rights_record")
    review = fixture.get("reference_review")
    if not isinstance(review, Mapping) or review.get("decision") not in {"approved_reference", "pending_reference_review"}:
        raise ValueError("real references require a saved approved_reference or explicit pending_reference_review state")
    if _digest(review.get("sha256")) != reference_sha:
        raise ValueError("human reference review is bound to a different artifact hash")
    if review["decision"] == "pending_reference_review":
        if set(review) != {"decision", "sha256", "reason"}:
            raise ValueError("pending references cannot claim reviewer identity or approval time")
        _text(review["reason"], "pending reference reason")
        return "pending_reference_review"
    _text(review.get("actor"), "reference reviewer")
    timestamp = datetime.fromisoformat(_text(review.get("reviewed_at"), "reviewed_at").replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("reference review timestamp requires a timezone")
    return "approved_reference"


def _crops(value: Any, width: int, height: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > 64:
        raise ValueError("register 1..64 native quality crops")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for crop in value:
        if not isinstance(crop, Mapping) or set(crop) != {"crop_id", "box", "concerns"}:
            raise ValueError("crop requires exactly crop_id, box and concerns")
        crop_id, box = _text(crop["crop_id"], "crop_id"), crop["box"]
        if crop_id in seen:
            raise ValueError("duplicate crop_id")
        seen.add(crop_id)
        if not isinstance(box, list) or len(box) != 4 or any(type(coordinate) is not int for coordinate in box):
            raise ValueError("crop box needs four integer native coordinates")
        left, top, right, bottom = box
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError("crop is outside the native image grid")
        result.append({"crop_id": crop_id, "box": box, "concerns": _labels(crop["concerns"], "crop concerns")})
    return result


def evaluate_property_evalset(evalset: Mapping[str, Any], *, asset_root: Path | str) -> dict[str, Any]:
    """Verify registered file hashes, then measure full frames and native crops.

    ``reference_review`` is a retained operator declaration, not independently
    authenticated identity or cryptographic approval. Candidate selection and
    approval remain separate human actions. Reports are deterministic for the
    same registrations, decoded samples and algorithm revision.
    """
    if not isinstance(evalset, Mapping) or evalset.get("schema") != EVALSET_SCHEMA:
        raise ValueError(f"evalset schema must be {EVALSET_SCHEMA}")
    evalset_id = _text(evalset.get("evalset_id"), "evalset_id")
    fixtures = evalset.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures or len(fixtures) > 256:
        raise ValueError("evalset must register 1..256 fixtures")
    root = Path(asset_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("asset_root must be an existing directory")
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    real_scenes: set[str] = set()
    real_concerns: set[str] = set()
    approved_scenes: set[str] = set()
    approved_concerns: set[str] = set()
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            raise ValueError("fixture must be an object")
        fixture_id = _text(fixture.get("fixture_id"), "fixture_id")
        if fixture_id in seen:
            raise ValueError("duplicate fixture_id")
        seen.add(fixture_id)
        scenes = _labels(fixture.get("scene_tags"), "scene_tags")
        group = _text(fixture.get("lighting_group"), "lighting_group")
        space = _text(fixture.get("working_color_space"), "working_color_space")
        transfer = _text(fixture.get("working_transfer_function"), "working_transfer_function")
        reference, reference_identity = _load_image(root, fixture.get("reference"))
        candidate, candidate_identity = _load_image(root, fixture.get("candidate"))
        reference_status = _validate_reference_review(fixture, reference_identity["sha256"])
        if reference.shape != candidate.shape:
            raise ValueError("register aligned reference/candidate images of identical dimensions")
        if reference_identity["icc_profile_sha256"] != candidate_identity["icc_profile_sha256"]:
            raise ValueError("register reference/candidate with matching embedded working profiles")
        crops = _crops(fixture.get("crops"), reference.shape[1], reference.shape[0])
        options = {"working_color_space": space, "working_transfer_function": transfer}
        crop_reports = []
        for crop in crops:
            left, top, right, bottom = crop["box"]
            crop_reports.append(
                {
                    **crop,
                    "scale": "native_1_to_1",
                    "metrics": compute_apex_metrics(
                        reference[top:bottom, left:right], candidate[top:bottom, left:right], **options
                    ),
                }
            )
        synthetic = fixture["synthetic_data"]
        if not synthetic:
            real_scenes.update(scenes)
            real_concerns.update(concern for crop in crops for concern in crop["concerns"])
        if reference_status == "approved_reference":
            approved_scenes.update(scenes)
            approved_concerns.update(concern for crop in crops for concern in crop["concerns"])
        reports.append(
            {
                "fixture_id": fixture_id,
                "scene_tags": scenes,
                "lighting_group": group,
                "synthetic_data": synthetic,
                "reference": reference_identity,
                "candidate": candidate_identity,
                "reference_review": fixture.get("reference_review"),
                "reference_status": reference_status,
                "rights_record": fixture.get("rights_record"),
                "full_frame": {"scale": "native_1_to_1", "metrics": compute_apex_metrics(reference, candidate, **options)},
                "native_crops": crop_reports,
                "human_candidate_approval": "not_assessed",
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "evalset_id": evalset_id,
        "algorithm_revision": ALGORITHM_REVISION,
        "photographic_production_acceptance": "not_assessed",
        "fixtures": reports,
        "real_reference_coverage": {
            "registered_scene_tags": sorted(real_scenes),
            "registered_crop_concerns": sorted(real_concerns),
            "missing_scene_tags": sorted(REQUIRED_SCENES - real_scenes),
            "missing_crop_concerns": sorted(REQUIRED_CONCERNS - real_concerns),
            "representativeness": "requires_independent_human_assessment",
        },
        "approved_reference_coverage": {
            "registered_scene_tags": sorted(approved_scenes),
            "registered_crop_concerns": sorted(approved_concerns),
            "missing_scene_tags": sorted(REQUIRED_SCENES - approved_scenes),
            "missing_crop_concerns": sorted(REQUIRED_CONCERNS - approved_concerns),
            "scope": "operator_declared_approved_references_only",
        },
        "limits": [
            "registered_same_geometry_and_encoding_required",
            "working_encoding_is_an_operator_assertion",
            "signals_are_not_calibrated_quality_gates",
            "synthetic_fixtures_do_not_count_as_real_coverage",
            "reference_review_is_an_operator_declaration",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Evaluate an external set; never overwrite an existing report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evalset", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        with args.evalset.open(encoding="utf-8") as handle:
            report = evaluate_property_evalset(json.load(handle), asset_root=args.asset_root)
        with args.output.open("x", encoding="utf-8") as handle:
            dump_json(report, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
