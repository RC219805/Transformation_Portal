"""Explicit native SDR Materials revisions; diagnostics never grant authority."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from PIL import Image

from transformation_portal.core.image_artifact import ImageMaster, validate_source_sha256
from transformation_portal.lux_depth_v4.photography import decode_master, linear_to_srgb, output_srgb_icc
from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES, encode_preview, preview_samples, preview_shape
from transformation_portal.lux_depth_v6.color import RenderRecipe
from transformation_portal.materials_v4.contracts import MaterialEvidence, RegionEvidence
from transformation_portal.materials_v4.engine import MaterialOperation, ResponsePolicy, apply_response, plan_response
from transformation_portal.materials_v4.operations import operation_contract_hash

from .models import DeliveryImage, bounded_number, canonical_bytes, sha256
from .presentation import RENDERER_CONTRACT

MATERIALS_RECIPE_SCHEMA = "tp.photo.materials_recipe.v1"
MATERIALS_REVISION_SCHEMA = "tp.photo.materials_revision.v1"
MAX_MATERIAL_REGIONS = 4
MAX_MATERIALS_DESCRIPTOR_BYTES = 256 * 1024
MAX_MATERIALS_OVERLAY_BYTES = MAX_PREVIEW_BYTES
_OPERATIONS = MappingProxyType({operation.label: operation.operation_id for operation in ResponsePolicy().operations})
MATERIAL_LABELS = frozenset(_OPERATIONS)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "processing_contract",
        "source_master_sha256",
        "source_identity_sha256",
        "source_bundle",
        "revision_sha256",
        "recipe",
        "recipe_sha256",
        "source_bit_depth",
        "output_bit_depth",
        "working_color",
        "output_color",
        "precision_recovery",
        "width",
        "height",
        "alpha_mode",
        "alpha_preserved",
        "input_master_content_sha256",
        "execution_output_master_content_sha256",
        "output_master_content_sha256",
        "evidence",
        "evidence_sha256",
        "policy",
        "policy_sha256",
        "operations_sha256",
        "execution",
        "preview",
        "preview_render",
        "output_presentation",
        "input_presentation_authority",
        "input_presentation_policy_sha256",
        "diagnostics",
        "review_status",
        "approval_transferred",
        "quality_acceptance",
    }
)


@dataclass(frozen=True)
class MaterialsRegion:
    id: str
    label: str
    region: tuple[float, float, float, float]
    strength: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _IDENTIFIER.fullmatch(self.id) is None:
            raise ValueError("Materials region requires a bounded ASCII identifier")
        if not isinstance(self.label, str) or self.label not in MATERIAL_LABELS:
            raise ValueError("Materials region requires a supported explicit label")
        if not isinstance(self.region, (tuple, list)) or len(self.region) != 4:
            raise ValueError("Materials region requires normalized left, top, right, bottom")
        region = tuple(bounded_number(value, "Materials coordinate", 0, 1) for value in self.region)
        if region[0] >= region[2] or region[1] >= region[3]:
            raise ValueError("Materials rectangle must have positive area")
        object.__setattr__(self, "region", region)
        object.__setattr__(self, "strength", bounded_number(self.strength, "Materials strength", 0, 0.05))

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "region": list(self.region), "strength": self.strength}


@dataclass(frozen=True)
class MaterialsRecipe:
    regions: tuple[MaterialsRegion, ...]
    input_stage: str = "display_referred"

    def __post_init__(self) -> None:
        if self.input_stage != "display_referred":
            raise ValueError("Native SDR Materials requires an explicit display-referred input assertion")
        if not isinstance(self.regions, (tuple, list)) or not 1 <= len(self.regions) <= MAX_MATERIAL_REGIONS:
            raise ValueError("Native Materials requires one to four explicit rectangles")
        if any(not isinstance(region, MaterialsRegion) for region in self.regions):
            raise ValueError("Native Materials requires validated explicit regions")
        regions = tuple(sorted(self.regions, key=lambda region: region.id))
        if len({region.id for region in regions}) != len(regions):
            raise ValueError("Materials region identifiers must be unique")
        strengths: dict[str, float] = {}
        for region in regions:
            if region.label in strengths and strengths[region.label] != region.strength:
                raise ValueError("Repeated Materials labels must share one explicit strength")
            strengths[region.label] = region.strength
        object.__setattr__(self, "regions", regions)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MaterialsRecipe:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema", "input_stage", "regions"}
            or payload["schema"] != MATERIALS_RECIPE_SCHEMA
            or not isinstance(payload["regions"], list)
            or not 1 <= len(payload["regions"]) <= MAX_MATERIAL_REGIONS
        ):
            raise ValueError("Native Materials recipe requires its exact closed schema")
        regions = []
        for region in payload["regions"]:
            if not isinstance(region, Mapping) or set(region) != {"id", "label", "region", "strength"}:
                raise ValueError("Materials region requires id, label, rectangle and strength")
            regions.append(MaterialsRegion(**dict(region)))
        return cls(tuple(regions), payload["input_stage"])

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": MATERIALS_RECIPE_SCHEMA,
            "input_stage": self.input_stage,
            "regions": [region.to_payload() for region in self.regions],
        }


def materials_policy(recipe: MaterialsRecipe) -> ResponsePolicy:
    """Freeze server-owned operation families and conservative execution limits."""
    strengths = {region.label: region.strength for region in recipe.regions}
    return ResponsePolicy(
        operations=tuple(MaterialOperation(label, _OPERATIONS[label], strength) for label, strength in strengths.items()),
        min_coverage_px=1,
        tile_size=256,
        max_abs_delta=0.025,
        max_working_bytes=128 * 1024**2,
        trusted_calibration_sha256=(),
    )


def _evidence(master: ImageMaster, recipe: MaterialsRecipe) -> MaterialEvidence:
    height, width = master.shape
    regions = []
    for region in recipe.regions:
        left, top, right, bottom = region.region
        x0, y0 = int(left * width), int(top * height)
        x1, y1 = max(x0 + 1, int(right * width)), max(y0 + 1, int(bottom * height))
        mask = np.zeros(master.shape, np.float32)
        mask[y0:y1, x0:x1] = 1
        regions.append(RegionEvidence(region.id, region.label, mask, semantic_confidence=1.0, provenance="supplied"))
    return MaterialEvidence(
        master.source_sha256,
        master.shape,
        tuple(regions),
        producer={
            "method": "explicit_human_native_rectangles_v1",
            "recipe_sha256": sha256(canonical_bytes(recipe.to_payload())),
        },
    )


def _tiff(master: ImageMaster) -> bytes:
    import tifffile

    # Reject extended samples rather than silently clipping a response to fit SDR.
    if not np.isfinite(master.pixels).all() or np.any((master.pixels < 0) | (master.pixels > 1)):
        raise ValueError("Native SDR Materials result must remain within unit linear RGB")
    samples = np.rint(linear_to_srgb(master.pixels) * 65535).astype(np.uint16)
    if master.alpha is not None:
        samples = np.concatenate([samples, np.rint(master.alpha * 65535).astype(np.uint16)[..., None]], axis=2)
    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer,
        samples,
        photometric="rgb",
        extrasamples="unassalpha" if master.alpha is not None else None,
        metadata={"color_space": "srgb", "color_domain": "display_encoded_srgb"},
        iccprofile=output_srgb_icc(),
        extratags=[(274, "H", 1, 1, False)],
    )
    return buffer.getvalue()


def _overlay(master: ImageMaster, evidence: MaterialEvidence) -> bytes:
    samples = preview_samples(master).copy()
    height, width = samples.shape[:2]
    coverage = np.zeros((height, width), np.float32)
    palette = {
        "glass": (40, 170, 255),
        "stone": (250, 180, 50),
        "water": (25, 90, 255),
        "foliage": (40, 210, 90),
        "sky": (180, 100, 255),
    }
    rgb = samples[..., :3].astype(np.float32)
    for region in evidence.regions:
        with Image.fromarray(region.mask) as mask:
            with mask.resize((width, height), Image.Resampling.BOX) as reduced:
                weight = np.asarray(reduced, dtype=np.float32)
        coverage += weight
        opacity = weight[..., None] * np.float32(0.35)
        rgb = rgb * (1 - opacity) + np.asarray(palette[region.label], np.float32) * opacity
    rgb[coverage > 1] = (255, 70, 70)
    samples[..., :3] = np.rint(rgb).astype(np.uint8)
    buffer = io.BytesIO()
    with Image.fromarray(samples) as image:
        image.save(buffer, format="PNG", icc_profile=output_srgb_icc(), optimize=False, compress_level=6)
    data = buffer.getvalue()
    if len(data) > MAX_MATERIALS_OVERLAY_BYTES:
        raise ValueError("Materials overlay exceeds its bounded derivative size")
    return data


def create_materials_revision(image: DeliveryImage, recipe: MaterialsRecipe) -> dict[str, Any]:
    """Create a separate SDR master and diagnostics, never an approval or dispatch."""
    if not isinstance(image, DeliveryImage) or not isinstance(recipe, MaterialsRecipe):
        raise ValueError("Native Materials requires an explicit photograph and validated recipe")
    if image.native is not None:
        raise ValueError("Native SDR Materials does not accept retained NPY working masters")
    if image.presentation is not None and image.presentation.stage != recipe.input_stage:
        raise ValueError("Materials input assertion conflicts with recorded photographic presentation")
    master = image.decode()
    if (
        master.source_bit_depth not in {8, 16}
        or not np.isfinite(master.pixels).all()
        or np.any((master.pixels < 0) | (master.pixels > 1))
    ):
        raise ValueError("Native Materials supports bounded 8/16-bit SDR photographic sources")
    evidence = _evidence(master, recipe)
    policy = materials_policy(recipe)
    plan = plan_response(master, evidence, policy)
    adjusted, execution = apply_response(master, evidence, plan)
    source_bytes = _tiff(adjusted)
    # Container quantization and new source metadata have their own exact identity.
    retained = decode_master(source_bytes, source_name="master.tif", input_color="auto")
    preview, preview_receipt = encode_preview(retained, relative_path="materials-preview.png")
    preview_receipt = {**preview_receipt, "sha256": sha256(preview), "size_bytes": len(preview)}
    overlay = _overlay(retained, evidence)
    render_recipe = image.presentation.recipe if image.presentation is not None else RenderRecipe(mode="clip_srgb")
    content_sha = retained.content_hash()
    preview_render = {
        "schema": "tp.photo.presentation_render.v1",
        "recipe": render_recipe.to_payload(),
        "operation": "preserve_recorded_display",
        "input_master_sha256": content_sha,
        "output_master_sha256": content_sha,
        "output_domain": "display_linear_srgb",
    }
    diagnostic = {
        "schema": "tp.photo.materials_diagnostics.v1",
        "diagnostic_only": True,
        "approval_authority": False,
        "recipe_sha256": sha256(canonical_bytes(recipe.to_payload())),
        "source_identity_sha256": image.source_identity_sha256,
        "input_master_content_sha256": master.content_hash(),
        "revision_sha256": sha256(source_bytes),
        "execution_output_master_content_sha256": adjusted.content_hash(),
        "output_master_content_sha256": content_sha,
        "evidence_sha256": evidence.content_hash(),
        "operations_sha256": operation_contract_hash(),
        "native_shape": list(master.shape),
        "preview": {
            **preview_receipt,
            "filename": "materials-preview.png",
            "sha256": sha256(preview),
            "size_bytes": len(preview),
            "meaning": "matching_retained_display_photograph",
        },
        "overlay": {
            "filename": "materials-overlay.png",
            "sha256": sha256(overlay),
            "size_bytes": len(overlay),
            "shape": list(preview_shape(master.shape)),
            "meaning": "human_rectangles_positive_overlap_red",
            "resampling": "area_average_native_mask",
            "source_coordinates": "orientation_normalized_native",
        },
        "execution": execution,
    }
    descriptor = canonical_bytes(diagnostic)
    if len(descriptor) > MAX_MATERIALS_DESCRIPTOR_BYTES:
        raise ValueError("Materials diagnostic descriptor exceeds its bounded size")
    receipt: dict[str, Any] = {
        "schema": MATERIALS_REVISION_SCHEMA,
        "processing_contract": "tp.photo.materials_revision_processing.v1",
        "source_master_sha256": image.source_sha256,
        "source_identity_sha256": image.source_identity_sha256,
        "source_bundle": image.source_bundle,
        "revision_sha256": sha256(source_bytes),
        "recipe": recipe.to_payload(),
        "recipe_sha256": sha256(canonical_bytes(recipe.to_payload())),
        "source_bit_depth": master.source_bit_depth,
        "output_bit_depth": 16,
        "working_color": "linear_srgb",
        "output_color": "display_encoded_srgb",
        "precision_recovery": False,
        "width": master.shape[1],
        "height": master.shape[0],
        "alpha_mode": "straight" if master.alpha is not None else None,
        "alpha_preserved": True,
        "input_master_content_sha256": master.content_hash(),
        "execution_output_master_content_sha256": adjusted.content_hash(),
        "output_master_content_sha256": content_sha,
        "evidence": evidence.to_payload(),
        "evidence_sha256": evidence.content_hash(),
        "policy": policy.to_payload(),
        "policy_sha256": sha256(canonical_bytes(policy.to_payload())),
        "operations_sha256": operation_contract_hash(),
        "execution": execution,
        "preview": preview_receipt,
        "preview_render": preview_render,
        "output_presentation": {
            "schema": "tp.photo.revision_presentation.v1",
            "stage": "display_referred",
            "recipe": render_recipe.to_payload(),
            "renderer_contract": RENDERER_CONTRACT,
        },
        "input_presentation_authority": "recorded_input_policy" if image.presentation else "explicit_display_sdr_assertion",
        "input_presentation_policy_sha256": image.presentation.sha256 if image.presentation else None,
        "diagnostics": [
            {"filename": name, "sha256": sha256(data), "size_bytes": len(data), "content_type": mime}
            for name, data, mime in (
                ("materials-overlay.png", overlay, "image/png"),
                ("materials-diagnostics.json", descriptor, "application/json"),
                ("materials-preview.png", preview, "image/png"),
            )
        ],
        "review_status": "unreviewed",
        "approval_transferred": False,
        "quality_acceptance": "human_review_required",
    }
    validate_materials_revision_receipt(receipt, receipt["revision_sha256"])
    return {
        "source_bytes": source_bytes,
        "preview_bytes": preview,
        "receipt": receipt,
        "diagnostic_files": {
            "materials-overlay.png": (overlay, "image/png"),
            "materials-diagnostics.json": (descriptor, "application/json"),
            "materials-preview.png": (preview, "image/png"),
        },
    }


def validate_materials_revision_receipt(receipt: Any, artifact_sha256: str) -> RenderRecipe:
    """Validate committed Materials presentation without granting storage access."""
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS or receipt.get("schema") != MATERIALS_REVISION_SCHEMA:
        raise ValueError("Unsupported native Materials revision receipt")
    recipe_payload = receipt.get("recipe")
    if not isinstance(recipe_payload, Mapping):
        raise ValueError("Materials receipt requires its closed recipe")
    recipe = MaterialsRecipe.from_payload(recipe_payload)
    if (
        receipt.get("revision_sha256") != artifact_sha256
        or receipt.get("processing_contract") != "tp.photo.materials_revision_processing.v1"
        or receipt.get("working_color") != "linear_srgb"
        or receipt.get("output_color") != "display_encoded_srgb"
        or type(receipt.get("output_bit_depth")) is not int
        or receipt["output_bit_depth"] != 16
        or type(receipt.get("source_bit_depth")) is not int
        or receipt["source_bit_depth"] not in {8, 16}
        or receipt.get("precision_recovery") is not False
        or receipt.get("alpha_mode") not in (None, "straight")
        or receipt.get("alpha_preserved") is not True
        or receipt.get("approval_transferred") is not False
        or receipt.get("review_status") != "unreviewed"
        or receipt.get("quality_acceptance") != "human_review_required"
    ):
        raise ValueError("Materials receipt does not bind its native SDR output contract")
    validate_source_sha256(artifact_sha256)
    for key in (
        "source_master_sha256",
        "source_identity_sha256",
        "input_master_content_sha256",
        "execution_output_master_content_sha256",
        "output_master_content_sha256",
    ):
        digest = receipt.get(key)
        if not isinstance(digest, str):
            raise ValueError("Materials receipt requires typed source and content identities")
        validate_source_sha256(digest)
    if receipt.get("source_bundle") != {
        "schema": "tp.photo.source_identity.v1",
        "master_sha256": receipt["source_master_sha256"],
        "descriptor_sha256": None,
        "alpha_sha256": None,
        "source_icc_sha256": None,
    } or (receipt["source_identity_sha256"] != receipt["source_master_sha256"]):
        raise ValueError("Materials receipt source identity is inconsistent")
    for key in ("width", "height"):
        if type(receipt.get(key)) is not int or not 1 <= receipt[key] <= 100_000_000:
            raise ValueError("Materials receipt has invalid native geometry")
    if receipt["width"] * receipt["height"] > 100_000_000:
        raise ValueError("Materials receipt exceeds native geometry limits")
    expected_policy = materials_policy(recipe).to_payload()
    if (
        receipt.get("recipe_sha256") != sha256(canonical_bytes(recipe.to_payload()))
        or receipt.get("policy") != expected_policy
        or receipt.get("policy_sha256") != sha256(canonical_bytes(expected_policy))
        or receipt.get("operations_sha256") != operation_contract_hash()
    ):
        raise ValueError("Materials receipt recipe, policy or operation identity changed")
    evidence = receipt.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("schema") != "tp.materials.evidence.v1"
        or evidence.get("source_sha256") != receipt["source_master_sha256"]
        or evidence.get("shape") != [receipt["height"], receipt["width"]]
        or evidence.get("calibration") is not None
        or evidence.get("status") != "available"
        or evidence.get("reason") is not None
        or evidence.get("coordinate_space") != "canonical_master"
        or evidence.get("producer")
        != {"method": "explicit_human_native_rectangles_v1", "recipe_sha256": receipt["recipe_sha256"]}
        or receipt.get("evidence_sha256") != sha256(canonical_bytes(evidence))
    ):
        raise ValueError("Materials receipt evidence identity changed")
    regions = evidence.get("regions")
    if not isinstance(regions, list) or len(regions) != len(recipe.regions):
        raise ValueError("Materials receipt lacks its explicit native masks")
    for recorded, region in zip(regions, recipe.regions):
        if (
            not isinstance(recorded, Mapping)
            or recorded.get("region_id") != region.id
            or recorded.get("label") != region.label
            or recorded.get("provenance") != "supplied"
            or recorded.get("semantic_confidence") != 1.0
            or recorded.get("calibration_sha256") is not None
            or recorded.get("geometric_quality") is not None
            or recorded.get("score_type") is not None
        ):
            raise ValueError("Materials receipt differs from human-supplied evidence")
        mask = recorded.get("mask")
        if (
            not isinstance(mask, Mapping)
            or set(mask) != {"shape", "dtype", "sha256"}
            or (mask["shape"] != evidence["shape"] or mask["dtype"] != "float32")
        ):
            raise ValueError("Materials receipt mask geometry changed")
        validate_source_sha256(mask["sha256"])
    execution = receipt.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Materials receipt lacks execution evidence")
    plan = execution.get("response_plan")
    if (
        not isinstance(plan, Mapping)
        or plan.get("schema") != "tp.materials.response_plan.v1"
        or plan.get("source_sha256") != receipt["source_master_sha256"]
        or plan.get("master_sha256") != receipt["input_master_content_sha256"]
        or plan.get("evidence_sha256") != receipt["evidence_sha256"]
        or plan.get("policy") != expected_policy
        or plan.get("operations_sha256") != receipt["operations_sha256"]
        or execution.get("schema") != "tp.materials.execution.v1"
        or execution.get("plan_sha256") != sha256(canonical_bytes(plan))
        or execution.get("input_master_sha256") != receipt["input_master_content_sha256"]
        or execution.get("output_master_sha256") != receipt["execution_output_master_content_sha256"]
        or execution.get("evidence_sha256") != receipt["evidence_sha256"]
        or execution.get("operations_sha256") != receipt["operations_sha256"]
        or execution.get("protected_changed_pixels") != 0
        or execution.get("protected_max_abs_delta") != 0
    ):
        raise ValueError("Materials execution differs from its frozen source, plan or protections")
    bounded_number(execution.get("max_abs_delta"), "executed Materials delta", 0, 0.025)
    bounded_number(execution.get("mean_abs_delta"), "mean Materials delta", 0, execution["max_abs_delta"])
    if type(execution.get("changed_pixels")) is not int or not 0 <= execution["changed_pixels"] <= (
        receipt["width"] * receipt["height"]
    ):
        raise ValueError("Materials execution has an invalid changed-pixel count")
    changed = execution["changed_pixels"]
    if (
        execution.get("status") != ("applied" if changed else "abstained")
        or execution.get("reason") != ("pixels_changed" if changed else "no_pixel_change")
        or execution.get("evidence_status") != "available"
        or execution.get("evidence_reason") is not None
    ):
        raise ValueError("Materials execution status conflicts with measured pixels")
    planned, measured = plan.get("regions"), execution.get("regions")
    if (
        not isinstance(planned, list)
        or not isinstance(measured, list)
        or len(planned) != len(recipe.regions)
        or len(measured) != len(recipe.regions)
    ):
        raise ValueError("Materials execution lacks its exact region decisions")
    region_changes = 0
    for proposed, actual, region in zip(planned, measured, recipe.regions):
        if (
            not isinstance(proposed, Mapping)
            or not isinstance(actual, Mapping)
            or set(proposed)
            != {"region_id", "label", "status", "reason", "coverage_px", "resolved_coverage_px", "operation_id", "strength"}
            or set(actual) != {*proposed, "changed_pixels"}
            or {key: value for key, value in actual.items() if key != "changed_pixels"} != proposed
            or proposed["region_id"] != region.id
            or proposed["label"] != region.label
            or proposed["status"] not in ("eligible", "abstained")
        ):
            raise ValueError("Materials region decision changed from its prepared response")
        coverage, resolved = proposed["coverage_px"], proposed["resolved_coverage_px"]
        if (
            type(coverage) is not int
            or type(resolved) is not int
            or not 0 <= resolved <= coverage <= receipt["width"] * receipt["height"]
            or type(actual["changed_pixels"]) is not int
            or not 0 <= actual["changed_pixels"] <= resolved
        ):
            raise ValueError("Materials region coverage or changed pixels are invalid")
        eligible = proposed["status"] == "eligible"
        if (
            proposed["operation_id"] != (_OPERATIONS[region.label] if eligible else None)
            or proposed["strength"] != (region.strength if eligible else None)
            or proposed["reason"] != ("evidence_accepted" if eligible else "below_resolved_coverage_threshold")
        ):
            raise ValueError("Materials region differs from its explicit bounded operation")
        region_changes += actual["changed_pixels"]
    if region_changes != changed:
        raise ValueError("Materials region changes disagree with actual execution totals")
    authority = receipt.get("input_presentation_authority")
    input_policy = receipt.get("input_presentation_policy_sha256")
    if authority == "recorded_input_policy":
        if not isinstance(input_policy, str):
            raise ValueError("Materials receipt requires a typed input presentation identity")
        validate_source_sha256(input_policy)
    elif authority != "explicit_display_sdr_assertion" or input_policy is not None:
        raise ValueError("Materials receipt lacks explicit input-stage authority")
    output = receipt.get("output_presentation")
    if (
        not isinstance(output, Mapping)
        or set(output) != {"schema", "stage", "recipe", "renderer_contract"}
        or output["schema"] != "tp.photo.revision_presentation.v1"
        or output["stage"] != "display_referred"
        or output["renderer_contract"] != RENDERER_CONTRACT
    ):
        raise ValueError("Materials receipt lacks its display-referred presentation")
    render_recipe = RenderRecipe.from_payload(output["recipe"])
    preview_render, preview = receipt.get("preview_render"), receipt.get("preview")
    if (
        not isinstance(preview_render, Mapping)
        or preview_render.get("schema") != "tp.photo.presentation_render.v1"
        or preview_render.get("operation") != "preserve_recorded_display"
        or preview_render.get("recipe") != render_recipe.to_payload()
        or preview_render.get("input_master_sha256") != receipt["output_master_content_sha256"]
        or preview_render.get("output_master_sha256") != receipt["output_master_content_sha256"]
        or preview_render.get("output_domain") != "display_linear_srgb"
        or not isinstance(preview, Mapping)
        or preview.get("schema") != "tp.image.browser_preview.v1"
        or preview.get("path") != "materials-preview.png"
        or preview.get("master_content_hash") != receipt["output_master_content_sha256"]
        or preview.get("shape") != list(preview_shape((receipt["height"], receipt["width"])))
        or preview.get("max_edge") != 1600
        or preview.get("output_bit_depth") != 8
        or preview.get("color_space") != "srgb"
        or preview.get("alpha_preserved") != (receipt["alpha_mode"] == "straight")
    ):
        raise ValueError("Materials preview must preserve its actual retained display pixels")
    diagnostics = receipt.get("diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 3:
        raise ValueError("Materials receipt lacks its bounded diagnostic products")
    for item, filename, mime, maximum in zip(
        diagnostics,
        ("materials-overlay.png", "materials-diagnostics.json", "materials-preview.png"),
        ("image/png", "application/json", "image/png"),
        (MAX_MATERIALS_OVERLAY_BYTES, MAX_MATERIALS_DESCRIPTOR_BYTES, MAX_PREVIEW_BYTES),
    ):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"filename", "sha256", "size_bytes", "content_type"}
            or item["filename"] != filename
            or item["content_type"] != mime
            or type(item["size_bytes"]) is not int
            or not 0 < item["size_bytes"] <= maximum
        ):
            raise ValueError("Materials diagnostic product identity is invalid")
        validate_source_sha256(item["sha256"])
    if preview.get("sha256") != diagnostics[2]["sha256"] or preview.get("size_bytes") != diagnostics[2]["size_bytes"]:
        raise ValueError("Materials saved preview differs from its matching photographic derivative")
    return render_recipe
