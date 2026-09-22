"""Deterministic V6 products shared by execution and semantic replay."""

from __future__ import annotations

import hashlib
import io
from typing import Any, Callable, Iterator

import numpy as np

from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.photography import linear_to_srgb, output_srgb_icc
from transformation_portal.lux_depth_v5.preview import encode_preview

from .color import GradeRecipe, RenderRecipe, grade_master, render_master
from .reconstruction import reconstruct_baseline
from .source import VerifiedV5Source, load_depth_inputs


def _delivery(master: ImageMaster, relative: str) -> tuple[bytes, dict[str, Any]]:
    import tifffile

    encoded = linear_to_srgb(master.pixels)
    samples = np.rint(np.clip(encoded, 0, 1) * 65535).astype(np.uint16)
    if master.alpha is not None:
        samples = np.concatenate([samples, np.rint(master.alpha * 65535).astype(np.uint16)[..., None]], axis=2)
    buffer = io.BytesIO()
    profile = output_srgb_icc()
    tifffile.imwrite(
        buffer,
        samples,
        photometric="rgb",
        extrasamples="unassalpha" if master.alpha is not None else None,
        metadata={
            "color_space": "srgb",
            "image_artifact_schema": "tp.image.master.v1",
            "color_domain": "display_encoded_srgb",
        },
        iccprofile=profile,
        extratags=[(274, "H", 1, 1, False)],
    )
    return buffer.getvalue(), {
        "path": relative,
        "bit_depth": 16,
        "color_space": "srgb",
        "domain": "display_encoded_srgb",
        "master_content_sha256": master.content_hash(),
        "icc_sha256": hashlib.sha256(profile).hexdigest(),
        "alpha_mode": "straight" if master.alpha is not None else None,
        "source_bit_depth": master.source_bit_depth,
    }


def image_products(
    source: VerifiedV5Source,
    input_id: str,
    grade: GradeRecipe,
    render: RenderRecipe,
    *,
    checkpoint: Callable[[], None],
) -> Iterator[tuple[str, bytes]]:
    """Reconstruct one photograph; retain unbounded grade separately from display."""
    from transformation_portal.core.execution_plan import decode_bounded_json_object

    checkpoint()
    original, depth, proxy = load_depth_inputs(source, input_id)
    configuration = decode_bounded_json_object(source.canonical_plan_bytes)["configuration"]
    baseline, reconstruction = reconstruct_baseline(original, depth, proxy, configuration)
    del original, depth, proxy
    checkpoint()
    graded, grade_receipt = grade_master(baseline, grade)
    checkpoint()
    display, render_receipt = render_master(graded, render)
    checkpoint()
    descriptor: dict[str, Any] = {
        "schema": "tp.lux.graded_photograph.v1",
        "input_id": input_id,
        "parent_source_digest": source.source_digest,
        "reconstruction": reconstruction,
        "baseline": baseline.to_payload(),
        "master": graded.to_payload(),
        "display": display.to_payload(),
        "grade": grade_receipt,
        "render": render_receipt,
        "production_acceptance": "not_established",
    }
    arrays = [("baseline.npy", baseline.pixels), ("master.npy", graded.pixels), ("display.npy", display.pixels)]
    if graded.alpha is not None:
        arrays.append(("alpha.npy", graded.alpha))
    for filename, array in arrays:
        checkpoint()
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        yield f"{input_id}/{filename}", buffer.getvalue()
    checkpoint()
    data, descriptor["delivery"] = _delivery(display, f"{input_id}/delivery.tif")
    yield f"{input_id}/delivery.tif", data
    del data
    checkpoint()
    data, descriptor["preview"] = encode_preview(display, relative_path=f"{input_id}/preview.png")
    yield f"{input_id}/preview.png", data
    del data
    yield f"{input_id}/photograph.json", canonicalize_json(descriptor)
