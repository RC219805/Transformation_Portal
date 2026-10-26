"""Recorded photographic presentation, separate from encoding and approval.

Only authenticated adapters establish descriptor/receipt authority. File names,
sample types, sample ranges and TIFF descriptions never establish this policy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from transformation_portal.core.image_artifact import ImageMaster, validate_source_sha256
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v6.color import RenderRecipe, render_master

RENDERER_CONTRACT = "tp.photo.presentation_renderer.v1"


@dataclass(frozen=True)
class PresentationPolicy:
    stage: str
    recipe: RenderRecipe
    authority_kind: str
    authority_sha256: str
    artifact_sha256: str
    source_identity_sha256: str
    input_master_sha256: str | None = None
    renderer_contract: str = RENDERER_CONTRACT

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage, str)
            or self.stage not in {"working_linear", "display_referred"}
            or not isinstance(self.recipe, RenderRecipe)
            or self.renderer_contract != RENDERER_CONTRACT
        ):
            raise ValueError("Unsupported recorded photographic presentation")
        if not isinstance(self.authority_kind, str) or self.authority_kind not in {
            "v6_working_master",
            "v6_delivery",
            "revision_v1",
            "revision_v2",
        }:
            raise ValueError("Unsupported photographic presentation authority")
        for digest in (self.authority_sha256, self.artifact_sha256, self.source_identity_sha256):
            validate_source_sha256(digest)
        if self.input_master_sha256 is not None:
            validate_source_sha256(self.input_master_sha256)
        if self.authority_kind == "v6_working_master" and self.input_master_sha256 is None:
            raise ValueError("Recorded native presentation requires its working content identity")
        if self.authority_kind == "v6_delivery" and self.stage != "display_referred":
            raise ValueError("A verified V6 delivery has already received its recorded render")
        if self.authority_kind in {"v6_working_master", "revision_v1"} and self.stage != "working_linear":
            raise ValueError("Recorded working-master authority cannot authorize a display-stage reinterpretation")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "tp.photo.presentation_policy.v1",
            **{key: value for key, value in self.__dict__.items() if key != "recipe"},
            "recipe": self.recipe.to_payload(),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonicalize_json(self.to_payload())).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PresentationPolicy:
        keys = {
            "schema",
            "stage",
            "recipe",
            "authority_kind",
            "authority_sha256",
            "artifact_sha256",
            "source_identity_sha256",
            "input_master_sha256",
            "renderer_contract",
        }
        if not isinstance(payload, Mapping) or set(payload) != keys or payload["schema"] != "tp.photo.presentation_policy.v1":
            raise ValueError("Unsupported recorded photographic presentation policy")
        return cls(
            **{key: value for key, value in payload.items() if key not in {"schema", "recipe"}},
            recipe=RenderRecipe.from_payload(payload["recipe"]),
        )

    def validate_binding(self, artifact_sha256: str, source_identity_sha256: str) -> None:
        if (self.artifact_sha256, self.source_identity_sha256) != (artifact_sha256, source_identity_sha256):
            raise ValueError("Recorded presentation belongs to different photographic bytes")

    def validate_master(self, master: ImageMaster) -> None:
        if self.authority_kind == "v6_working_master":
            native = master.metadata.get("native_working_master", {})
            if native.get("retained_descriptor_content_sha256") != self.input_master_sha256:
                raise ValueError("Recorded presentation differs from the retained native working content")


def _render_receipt(payload: Any) -> RenderRecipe:
    if not isinstance(payload, Mapping) or payload.get("schema") != "tp.lux.sdr_render_receipt.v1":
        raise ValueError("Recorded presentation requires a recognized SDR render receipt")
    if payload.get("output_domain") != "display_linear_srgb":
        raise ValueError("Recorded render has an unsupported output domain")
    for key in ("input_master_sha256", "output_master_sha256"):
        digest = payload.get(key)
        if not isinstance(digest, str):
            raise ValueError("Recorded render lacks a content identity")
        validate_source_sha256(digest)
    recipe = payload.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("Recorded render lacks a typed recipe")
    return RenderRecipe.from_payload(recipe)


def descriptor_policy(
    descriptor: Mapping[str, Any],
    *,
    descriptor_sha256: str,
    artifact_sha256: str,
    source_identity_sha256: str,
    delivery_path: str | None = None,
) -> PresentationPolicy | None:
    """Resolve only recognized V6 semantics from an already verified descriptor."""
    if descriptor.get("schema") not in ("tp.lux.graded_photograph.v1", "tp.lux.graded_photograph.v2"):
        return None
    if not {"render", "display", "delivery"} <= descriptor.keys():
        return None
    render = descriptor.get("render")
    if not isinstance(render, Mapping):
        raise ValueError("V6 descriptor lacks a typed render receipt")
    recipe = _render_receipt(render)
    display = descriptor.get("display")
    if (
        not isinstance(display, Mapping)
        or display.get("schema") != "tp.image.master.v1"
        or display.get("color_space") != "linear_srgb"
        or not isinstance(display.get("metadata"), Mapping)
        or display["metadata"].get("color_domain") != "display_linear_srgb"
        or display["metadata"].get("display_render") != recipe.to_payload()
    ):
        raise ValueError("V6 descriptor lacks matching recorded display authority")
    if delivery_path is not None:
        delivery = descriptor.get("delivery")
        if (
            not isinstance(delivery, Mapping)
            or delivery.get("path") != delivery_path
            or delivery.get("domain") != "display_encoded_srgb"
            or delivery.get("color_space") != "srgb"
            or delivery.get("master_content_sha256") != render["output_master_sha256"]
        ):
            raise ValueError("V6 delivery and recorded render authority disagree")
    return PresentationPolicy(
        "display_referred" if delivery_path is not None else "working_linear",
        recipe,
        "v6_delivery" if delivery_path is not None else "v6_working_master",
        descriptor_sha256,
        artifact_sha256,
        source_identity_sha256,
        None if delivery_path is not None else render["input_master_sha256"],
    )


def revision_policy(receipt: Any, artifact_sha256: str) -> PresentationPolicy | None:
    """Recover known historical semantics; unknown receipts never grant a view."""
    if not isinstance(receipt, Mapping) or receipt.get("schema") not in ("tp.photo.revision.v1", "tp.photo.revision.v2"):
        return None
    if (
        receipt.get("revision_sha256") != artifact_sha256
        or receipt.get("output_color") != "extended_encoded_srgb"
        or receipt.get("output_bit_depth") != 32
        or receipt.get("working_color") != "extended_linear_srgb"
    ):
        raise ValueError("Saved revision receipt does not bind its recorded photographic bytes")
    if receipt["schema"] == "tp.photo.revision.v1":
        # V1 always wrote the unbounded adjusted master and rendered its preview
        # once. Preserve that historical contract, even for a delivery parent.
        recipe = _render_receipt(receipt.get("preview_render"))
        stage, kind = "working_linear", "revision_v1"
    else:
        if receipt.get("processing_contract") != "tp.photo.revision_processing.v2" or receipt.get(
            "input_presentation_authority"
        ) not in ("recorded_input_policy", "explicit_legacy_revision_operation"):
            raise ValueError("Saved revision lacks its recorded processing contract")
        input_policy = receipt.get("input_presentation_policy_sha256")
        if receipt["input_presentation_authority"] == "recorded_input_policy":
            if not isinstance(input_policy, str):
                raise ValueError("Saved revision lacks its recorded input policy identity")
            validate_source_sha256(input_policy)
        elif input_policy is not None:
            raise ValueError("A legacy revision operation cannot invent input presentation authority")
        output = receipt.get("output_presentation")
        if (
            not isinstance(output, Mapping)
            or set(output) != {"schema", "stage", "recipe", "renderer_contract"}
            or output["schema"] != "tp.photo.revision_presentation.v1"
        ):
            raise ValueError("Saved revision lacks explicit output presentation")
        stage, kind = output["stage"], "revision_v2"
        if receipt["input_presentation_authority"] == "explicit_legacy_revision_operation" and stage != "working_linear":
            raise ValueError("A legacy revision operation only records its actual working-master output")
        recipe = RenderRecipe.from_payload(output["recipe"])
        if output["renderer_contract"] != RENDERER_CONTRACT:
            raise ValueError("Saved revision requires its historical presentation renderer")
    preview_render = receipt.get("preview_render")
    output_content = receipt.get("output_master_content_sha256")
    if not isinstance(output_content, str):
        raise ValueError("Saved revision lacks its output content identity")
    validate_source_sha256(output_content)
    if not isinstance(preview_render, Mapping) or preview_render.get("recipe") != recipe.to_payload():
        raise ValueError("Saved revision and preview presentation recipes disagree")
    if preview_render.get("input_master_sha256") != receipt.get("output_master_content_sha256"):
        raise ValueError("Saved revision preview belongs to different working pixels")
    if stage == "working_linear":
        _render_receipt(preview_render)
    elif (
        preview_render.get("schema") != "tp.photo.presentation_render.v1"
        or preview_render.get("operation") != "preserve_recorded_display"
        or preview_render.get("output_master_sha256") != receipt.get("output_master_content_sha256")
    ):
        raise ValueError("Saved display-referred revision has conflicting preview processing")
    return PresentationPolicy(
        stage, recipe, kind, hashlib.sha256(canonicalize_json(receipt)).hexdigest(), artifact_sha256, artifact_sha256
    )


def render_presentation(
    master: ImageMaster,
    policy: PresentationPolicy | None,
    recipe: RenderRecipe,
) -> tuple[ImageMaster, dict[str, Any]]:
    """Render working pixels once; display-referred pixels only need encoding."""
    if policy is None:
        return render_master(master, recipe)
    if policy.stage == "working_linear":
        result, receipt = render_master(master, policy.recipe)
        return result, {**receipt, "requested_recipe": recipe.to_payload(), "presentation_policy_sha256": policy.sha256}
    return master, {
        "schema": "tp.photo.presentation_render.v1",
        "recipe": policy.recipe.to_payload(),
        "requested_recipe": recipe.to_payload(),
        "operation": "preserve_recorded_display",
        "presentation_policy_sha256": policy.sha256,
        "input_master_sha256": master.content_hash(),
        "output_master_sha256": master.content_hash(),
        "output_domain": "display_linear_srgb",
        "quality_acceptance": "unestablished",
    }
