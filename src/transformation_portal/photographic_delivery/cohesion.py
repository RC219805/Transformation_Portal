"""Explicit, reversible set-cohesion revisions; diagnostics never auto-match."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from transformation_portal.core.image_artifact import ImageMaster, metadata_payload, validate_source_sha256
from transformation_portal.lux_depth_v4.photography import linear_to_srgb, output_srgb_icc
from transformation_portal.lux_depth_v5.preview import encode_preview
from transformation_portal.lux_depth_v6.color import GradeRecipe, RenderRecipe, grade_master

from .models import DeliveryImage, bounded_number, canonical_bytes, sha256, slug
from .presentation import RENDERER_CONTRACT, render_presentation


@dataclass(frozen=True)
class MaterialAdjustment:
    """Human-labelled normalized rectangular material region with feathered edges."""

    name: str
    region: tuple[float, float, float, float]
    exposure_stops: float = 0.0
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    feather: float = 0.02

    def __post_init__(self) -> None:
        slug(self.name, "material name")
        if not isinstance(self.region, (tuple, list)) or len(self.region) != 4:
            raise ValueError("Material region must contain normalized left, top, right, bottom")
        region = tuple(bounded_number(value, "material coordinate", 0, 1) for value in self.region)
        if not region[0] < region[2] or not region[1] < region[3]:
            raise ValueError("Material region must have positive area")
        object.__setattr__(self, "region", region)
        object.__setattr__(self, "exposure_stops", bounded_number(self.exposure_stops, "material exposure_stops", -0.5, 0.5))
        if not isinstance(self.white_balance, (tuple, list)) or len(self.white_balance) != 3:
            raise ValueError("Material white balance requires three explicit RGB gains")
        object.__setattr__(
            self, "white_balance", tuple(bounded_number(v, "material RGB gain", 0.9, 1.1) for v in self.white_balance)
        )
        object.__setattr__(self, "feather", bounded_number(self.feather, "material feather", 0.005, 0.1))

    def to_payload(self) -> dict[str, Any]:
        return {**self.__dict__, "region": list(self.region), "white_balance": list(self.white_balance)}


@dataclass(frozen=True)
class RevisionRecipe:
    exposure_stops: float = 0.0
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    saturation: float = 1.0
    materials: tuple[MaterialAdjustment, ...] = ()
    reference_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "exposure_stops", bounded_number(self.exposure_stops, "exposure_stops", -1, 1))
        object.__setattr__(self, "saturation", bounded_number(self.saturation, "saturation", 0.8, 1.2))
        if not isinstance(self.white_balance, (tuple, list)) or len(self.white_balance) != 3:
            raise ValueError("White balance requires three explicit RGB gains")
        object.__setattr__(self, "white_balance", tuple(bounded_number(v, "RGB gain", 0.8, 1.25) for v in self.white_balance))
        if not isinstance(self.materials, (tuple, list)) or len(self.materials) > 16:
            raise ValueError("At most sixteen explicit material adjustments are supported")
        if any(not isinstance(item, MaterialAdjustment) for item in self.materials):
            raise ValueError("Material adjustments require explicit validated regions")
        names = [item.name.casefold() for item in self.materials]
        if len(set(names)) != len(names):
            raise ValueError("Material adjustment names must be unique")
        object.__setattr__(self, "materials", tuple(self.materials))
        if self.reference_sha256 is not None:
            validate_source_sha256(self.reference_sha256)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "tp.photo.revision_recipe.v1",
            "exposure_stops": self.exposure_stops,
            "white_balance": list(self.white_balance),
            "saturation": self.saturation,
            "materials": [item.to_payload() for item in self.materials],
            "reference_sha256": self.reference_sha256,
            "automatic_matching": False,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RevisionRecipe:
        if (
            not isinstance(payload, Mapping)
            or set(payload)
            - {
                "schema",
                "exposure_stops",
                "white_balance",
                "saturation",
                "materials",
                "reference_sha256",
                "automatic_matching",
            }
            or payload.get("schema", "tp.photo.revision_recipe.v1") != "tp.photo.revision_recipe.v1"
        ):
            raise ValueError("Unsupported photographic revision recipe")
        if payload.get("automatic_matching", False) is not False:
            raise ValueError("Automatic set matching is forbidden; choose explicit adjustments")
        values = {key: value for key, value in payload.items() if key not in {"schema", "automatic_matching"}}
        materials = values.get("materials", [])
        if not isinstance(materials, (tuple, list)) or len(materials) > 16:
            raise ValueError("Material adjustments must be a bounded list")
        parsed = []
        for item in materials:
            if not isinstance(item, Mapping) or set(item) - {"name", "region", "exposure_stops", "white_balance", "feather"}:
                raise ValueError("Unsupported material adjustment")
            try:
                parsed.append(MaterialAdjustment(**item))
            except TypeError as error:
                raise ValueError("Material adjustment requires name and region") from error
        values["materials"] = tuple(parsed)
        return cls(**values)


def extended_tiff_bytes(master: ImageMaster) -> bytes:
    """Float32 encoded-sRGB TIFF, preserving finite extended values and alpha."""
    import tifffile

    pixels = linear_to_srgb(master.pixels)
    if not np.isfinite(pixels).all():
        raise ValueError("Source cannot be represented by a finite float32 sRGB TIFF")
    if master.alpha is not None:
        pixels = np.concatenate([pixels, master.alpha[..., None]], axis=2)
    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer,
        pixels.astype(np.float32),
        photometric="rgb",
        extrasamples="unassalpha" if master.alpha is not None else None,
        metadata={"color_space": "srgb", "color_domain": "extended_encoded_srgb", "precision": "float32"},
        iccprofile=output_srgb_icc(),
        extratags=[(274, "H", 1, 1, False)],
    )
    return buffer.getvalue()


def _apply_materials(master: ImageMaster, materials: tuple[MaterialAdjustment, ...]) -> ImageMaster:
    if not materials:
        return master
    height, width = master.shape
    pixels = master.pixels.copy()
    opaque = np.ones((height, width), dtype=bool) if master.alpha is None else master.alpha == 1
    # Overlapping user regions combine in log-gain space; bound the total so
    # sixteen overlapping rectangles cannot defeat per-image adjustment limits.
    log_gain = np.zeros_like(pixels)
    for material in materials:
        left, top, right, bottom = material.region
        region = np.zeros((height, width), dtype=np.float32)
        region[
            int(top * height) : max(int(top * height) + 1, int(bottom * height)),
            int(left * width) : max(int(left * width) + 1, int(right * width)),
        ] = 1
        mask = gaussian_filter(region, sigma=max(0.5, material.feather * min(height, width)), mode="constant")
        gain = material.exposure_stops + np.log2(np.asarray(material.white_balance, dtype=np.float32))
        log_gain += mask[..., None] * gain
    log_gain = np.clip(log_gain, -0.75, 0.75)
    pixels[opaque] *= np.exp2(log_gain[opaque])
    metadata = metadata_payload(master.metadata)
    metadata["material_revision"] = [item.to_payload() for item in materials]
    return ImageMaster(pixels, master.source_sha256, master.source_bit_depth, master.alpha, metadata, master.source_icc)


def create_revision(image: DeliveryImage, recipe: RevisionRecipe, *, reference_sha256: str | None = None) -> dict[str, Any]:
    """Create separate precise revision bytes and a bounded preview, never approval."""
    if not isinstance(recipe, RevisionRecipe):
        raise ValueError("Revision requires a validated explicit recipe")
    if reference_sha256 is not None:
        validate_source_sha256(reference_sha256)
        if recipe.reference_sha256 not in {None, reference_sha256}:
            raise ValueError("Reference identity differs from the recorded recipe")
        recipe = RevisionRecipe(
            recipe.exposure_stops, recipe.white_balance, recipe.saturation, recipe.materials, reference_sha256
        )
    master = image.decode()
    adjusted, grade_receipt = grade_master(
        master,
        GradeRecipe(exposure_stops=recipe.exposure_stops, white_balance=recipe.white_balance, saturation=recipe.saturation),
    )
    adjusted = _apply_materials(adjusted, recipe.materials)
    source_bytes = extended_tiff_bytes(adjusted)
    render_recipe = image.presentation.recipe if image.presentation is not None else RenderRecipe()
    display, render_receipt = render_presentation(adjusted, image.presentation, render_recipe)
    preview, preview_receipt = encode_preview(display, relative_path="preview.png")
    payload = recipe.to_payload()
    receipt = {
        "schema": "tp.photo.revision.v2",
        "processing_contract": "tp.photo.revision_processing.v2",
        "source_master_sha256": image.source_sha256,
        "source_identity_sha256": image.source_identity_sha256,
        "source_bundle": image.source_bundle,
        "revision_sha256": sha256(source_bytes),
        "recipe": payload,
        "recipe_sha256": sha256(canonical_bytes(payload)),
        "input_master_content_sha256": master.content_hash(),
        "output_master_content_sha256": adjusted.content_hash(),
        "working_color": "extended_linear_srgb",
        "output_color": "extended_encoded_srgb",
        "output_bit_depth": 32,
        "source_bit_depth": master.source_bit_depth,
        "precision_recovery": False,
        "width": master.shape[1],
        "height": master.shape[0],
        "reference_sha256": recipe.reference_sha256,
        "grade": grade_receipt,
        "preview_render": render_receipt,
        "preview": preview_receipt,
        "output_presentation": {
            "schema": "tp.photo.revision_presentation.v1",
            "stage": image.presentation.stage if image.presentation is not None else "working_linear",
            "recipe": render_recipe.to_payload(),
            "renderer_contract": RENDERER_CONTRACT,
        },
        "input_presentation_policy_sha256": image.presentation.sha256 if image.presentation is not None else None,
        "input_presentation_authority": (
            "recorded_input_policy" if image.presentation is not None else "explicit_legacy_revision_operation"
        ),
        "review_status": "unreviewed",
        "approval_transferred": False,
        "quality_acceptance": "human_review_required",
    }
    return {"source_bytes": source_bytes, "preview_bytes": preview, "receipt": receipt}


def cohesion_diagnostics(images: Sequence[DeliveryImage], lighting_groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reference-relative statistics only; composition/intent can explain outliers."""
    if (
        not images
        or len(images) > 100
        or any(not isinstance(image, DeliveryImage) for image in images)
        or not isinstance(lighting_groups, (tuple, list))
        or not 1 <= len(lighting_groups) <= 100
    ):
        raise ValueError("Cohesion review requires one to one hundred photographs and lighting groups")
    stats = {}
    for image in images:
        if image.source_identity_sha256 in stats:
            raise ValueError("Cohesion inputs require unique exact master hashes")
        master = image.decode()
        values = master.pixels.reshape(-1, 3)
        if master.alpha is not None:
            values = values[master.alpha.reshape(-1) == 1]
        if not len(values):
            raise ValueError("Cohesion diagnostics require opaque photographic samples")
        rgb = np.median(np.maximum(values.astype(np.float64), 0), axis=0)
        luminance = float(np.median(np.maximum(values @ np.array([0.2126, 0.7152, 0.0722]), 0)))
        chroma = rgb / max(float(rgb.sum()), 1e-9)
        stats[image.source_identity_sha256] = {
            "source_master_sha256": image.source_sha256,
            "source_identity_sha256": image.source_identity_sha256,
            "source_bundle": image.source_bundle,
            "scene": image.scene,
            "median_luminance": luminance,
            "rgb_chromaticity": chroma.tolist(),
        }
    seen = set()
    groups = []
    group_names = set()
    for group in lighting_groups:
        if not isinstance(group, Mapping) or set(group) != {"name", "reference_sha256", "members"}:
            raise ValueError("Lighting groups require name, reference_sha256, and exact member hashes")
        name = slug(group["name"], "lighting group")
        reference = group["reference_sha256"]
        validate_source_sha256(reference)
        members = group["members"]
        if name in group_names or not isinstance(members, (tuple, list)) or not members or reference not in members:
            raise ValueError("Lighting group needs a unique name and an included explicit reference photograph")
        group_names.add(name)
        rows = []
        for member in members:
            if not isinstance(member, str) or member not in stats or member in seen:
                raise ValueError("Lighting groups must partition known master hashes exactly once")
            seen.add(member)
            delta_ev = float(
                np.log2((stats[member]["median_luminance"] + 1e-6) / (stats[reference]["median_luminance"] + 1e-6))
            )
            color_delta = float(
                np.max(np.abs(np.array(stats[member]["rgb_chromaticity"]) - stats[reference]["rgb_chromaticity"]))
            )
            flags = []
            if abs(delta_ev) > 0.75:
                flags.append("exposure_outlier")
            if color_delta > 0.12:
                flags.append("chromaticity_outlier")
            rows.append(
                {
                    **stats[member],
                    "exposure_delta_ev": delta_ev,
                    "chromaticity_delta": color_delta,
                    "flags": flags,
                }
            )
        groups.append({"name": group["name"], "reference_sha256": reference, "images": rows})
    if seen != set(stats):
        raise ValueError("Every photograph must belong to one explicit lighting group")
    return {
        "schema": "tp.photo.cohesion_diagnostics.v1",
        "groups": groups,
        "automatic_adjustment": False,
        "exposure_outlier_threshold_ev": 0.75,
        "chromaticity_outlier_threshold": 0.12,
        "interpretation": "Advisory composition-dependent statistics; preserve intentional warmth and twilight; human review required",
    }
