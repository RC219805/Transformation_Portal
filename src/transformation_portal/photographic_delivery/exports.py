"""Named destination exports with source linkage and merged verified packaging."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from transformation_portal.core.image_artifact import ImageMaster, metadata_payload
from transformation_portal.core.photographic_export import jpeg_bytes, resize_long_edge, unsharp_mask
from transformation_portal.lux_depth_v4.photography import linear_to_srgb, output_srgb_icc
from transformation_portal.lux_depth_v6.color import RenderRecipe, render_master

from .cohesion import extended_tiff_bytes
from .models import DeliveryImage, ExportPreset, canonical_bytes, sha256, slug


def _safe_empty_output(output_dir: Path) -> Path:
    path = Path(output_dir).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Delivery output must not contain symlinks")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("Delivery output must be a fresh or empty directory")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _write_new(root: Path, relative: str, data: bytes) -> dict[str, Any]:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(data)
    return {"path": relative, "sha256": sha256(data), "size_bytes": len(data)}


def _sdr(
    master: ImageMaster, preset: ExportPreset, display: ImageMaster, receipt: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    if master.alpha is not None and np.any(master.alpha != 1):
        raise ValueError("SDR export of transparency requires an explicit external background composite")
    pixels = display.pixels
    if preset.long_edge is not None:
        pixels = resize_long_edge(pixels, preset.long_edge)
    pixels = unsharp_mask(pixels, amount=preset.sharpen_amount)
    return pixels, receipt


def _tiff16(pixels: np.ndarray, metadata: dict[str, str]) -> bytes:
    import tifffile

    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer,
        np.rint(np.clip(linear_to_srgb(pixels), 0, 1) * 65535).astype(np.uint16),
        photometric="rgb",
        metadata={"color_space": "srgb", "color_domain": "display_encoded_srgb", **metadata},
        iccprofile=output_srgb_icc(),
        extratags=[(274, "H", 1, 1, False)],
    )
    return buffer.getvalue()


def _contact_sheet(previews: list[tuple[str, np.ndarray]]) -> bytes:
    columns = min(4, len(previews))
    rows = (len(previews) + columns - 1) // columns
    cell_width, cell_height = 360, 290
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), (242, 242, 242))
    drawing = ImageDraw.Draw(canvas)
    for index, (label, encoded) in enumerate(previews):
        image = Image.fromarray(np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8))
        image.thumbnail((cell_width - 20, cell_height - 60), Image.Resampling.LANCZOS)
        x, y = index % columns * cell_width, index // columns * cell_height
        canvas.paste(image, (x + (cell_width - image.width) // 2, y + 10))
        drawing.text((x + 10, y + cell_height - 40), label[:48], fill=(20, 20, 20))
    return jpeg_bytes(np.asarray(canvas).astype(np.float32) / 255, icc_profile=output_srgb_icc(), quality=92)


def export_delivery(
    images: Sequence[DeliveryImage], output_dir: Path, *, property_name: str, export_presets: Sequence[ExportPreset]
) -> dict[str, Any]:
    """Export one caller-approved set, retaining exact source hashes in every row.

    This creates new bytes only. It does not implement authentication or infer
    approval; managed callers must require approved decisions bound to each
    source hash. A failed operation may leave its private incomplete directory,
    never replacing an existing delivery.
    """
    if not images or len(images) > 100 or any(not isinstance(image, DeliveryImage) for image in images):
        raise ValueError("Delivery requires one to one hundred explicit photographic masters")
    presets = tuple(export_presets)
    if not presets or len(presets) > 6 or any(not isinstance(preset, ExportPreset) for preset in presets):
        raise ValueError("Delivery requires explicit validated destination presets")
    if len({preset.name for preset in presets}) != len(presets):
        raise ValueError("Destination presets must be unique")
    if len({image.source_identity_sha256 for image in images}) != len(images):
        raise ValueError("Approved delivery entries require unique exact master hashes")
    property_slug = slug(property_name, "property_name")
    root = _safe_empty_output(output_dir)
    records, previews, sources, contact_sources, native_components = [], [], [], [], []
    for image in images:
        data = image.snapshot()
        master = image.decode(data)
        stem = f"{property_slug}-{slug(image.scene, 'scene')}-{slug(image.version, 'version')}-{image.source_identity_sha256[:10]}"
        source_info = {
            "source_master_sha256": image.source_sha256,
            "source_identity_sha256": image.source_identity_sha256,
            "source_bundle": image.source_bundle,
            "source_name": image.source_name,
            "source_format": master.metadata["source_format"],
            "scene": image.scene,
            "version": image.version,
            "width": master.shape[1],
            "height": master.shape[0],
            "bit_depth": master.source_bit_depth,
            "working_color": "extended_linear_srgb",
            "source_color": master.metadata["input_color"],
            "color_preparation": metadata_payload(master.metadata["color_preparation"]),
            "source_icc_sha256": sha256(master.source_icc) if master.source_icc else None,
            "master_content_sha256": master.content_hash(),
            "metadata": dict(image.metadata),
            **metadata_payload(master.metadata.get("native_working_master", {})),
        }
        sources.append(source_info)
        cached_render = None
        cached_display = None
        cached_receipt = None
        contact_preset = None
        contact_derivative_sha256 = None
        display: np.ndarray | None = None
        for preset in presets:
            recipe = preset.to_payload()
            receipt = None
            color_profile = "srgb"
            if preset.format == "source":
                extension = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "NPY": ".npy"}[master.metadata["source_format"]]
                payload = data
                height, width, bits = *master.shape, master.source_bit_depth
                color_profile = master.metadata["input_color"]
                profile_sha = source_info["source_icc_sha256"]
                domain = "extended_linear_srgb" if image.native is not None else "source_encoding_preserved"
            elif preset.format == "tiff32":
                payload, extension = extended_tiff_bytes(master), ".tif"
                height, width, bits = *master.shape, 32
                profile_sha = sha256(output_srgb_icc())
                domain = "extended_encoded_srgb"
            else:
                render_key = (preset.render, preset.shoulder)
                if cached_render != render_key or cached_display is None or cached_receipt is None:
                    cached_display, cached_receipt = render_master(master, RenderRecipe(*render_key))
                    cached_render = render_key
                display, receipt = _sdr(master, preset, cached_display, cached_receipt)
                height, width = display.shape[:2]
                profile_sha = sha256(output_srgb_icc())
                domain = "display_encoded_srgb"
                if preset.format == "tiff16":
                    payload, extension, bits = _tiff16(display, dict(image.metadata)), ".tif", 16
                else:
                    encoded = np.clip(linear_to_srgb(display), 0, 1)
                    payload, extension, bits = (
                        jpeg_bytes(
                            encoded, icc_profile=output_srgb_icc(), quality=preset.jpeg_quality, metadata=image.metadata
                        ),
                        ".jpg",
                        8,
                    )
            relative = f"{preset.name}/{stem}-{preset.name}{extension}"
            record = {
                **_write_new(root, relative, payload),
                "source_master_sha256": image.source_sha256,
                "source_identity_sha256": image.source_identity_sha256,
                "source_bundle": image.source_bundle,
                "source_master_content_sha256": source_info["master_content_sha256"],
                "source_bit_depth": master.source_bit_depth,
                "precision_recovery": False,
                "container_promotion": bits > master.source_bit_depth,
                "preset": preset.name,
                "recipe": recipe,
                "recipe_sha256": sha256(canonical_bytes(recipe)),
                "width": width,
                "height": height,
                "bit_depth": bits,
                "color_profile": color_profile,
                "icc_sha256": profile_sha,
                "color_domain": domain,
                "intended_use": preset.intended_use,
                "render": receipt,
                "approval_transferred": False,
            }
            records.append(record)
            if preset.format == "source" and image.native is not None:
                # Preserve the exact array and all selected authorizing components.
                # A separate usable TIFF transfers extended values analytically;
                # it never substitutes for the retained linear array authority.
                for name, component in image.native.component_snapshots().items():
                    component_suffix = {"descriptor": "photograph.json", "alpha": "alpha.npy", "source_icc": "source.icc"}[
                        name
                    ]
                    native_components.append(
                        {
                            **_write_new(root, f"photographic_master/{stem}-{component_suffix}", component),
                            "component": name,
                            "source_master_sha256": image.source_sha256,
                            "source_identity_sha256": image.source_identity_sha256,
                            "source_bundle": image.source_bundle,
                            "approval_transferred": False,
                        }
                    )
                precision = extended_tiff_bytes(master)
                records.append(
                    {
                        **record,
                        **_write_new(root, f"photographic_master/{stem}-precision.tif", precision),
                        "color_profile": "srgb",
                        "color_domain": "extended_encoded_srgb",
                        "icc_sha256": sha256(output_srgb_icc()),
                        "intended_use": "Float32 retouch TIFF companion; retained NPY and descriptor remain precision authority",
                        "recipe": {**recipe, "companion": "analytic_extended_srgb_transfer_no_clipping"},
                        "recipe_sha256": sha256(
                            canonical_bytes({**recipe, "companion": "analytic_extended_srgb_transfer_no_clipping"})
                        ),
                    }
                )
                del precision
            if preset.format in {"jpeg", "tiff16"}:
                contact_preset = preset.name
                contact_derivative_sha256 = record["sha256"]
            del payload
        if cached_display is not None:
            preview_master, preview_render = cached_display, cached_receipt
        else:
            preview_master, preview_render = render_master(master, RenderRecipe())
        preview_pixels = resize_long_edge(preview_master.pixels, 340)
        if preview_master.alpha is not None and np.any(preview_master.alpha != 1):
            # Contact-sheet-only white composite is explicit; master alpha stays intact.
            alpha = Image.fromarray(preview_master.alpha).resize(
                (preview_pixels.shape[1], preview_pixels.shape[0]), Image.Resampling.LANCZOS
            )
            alpha_array = np.clip(np.asarray(alpha), 0, 1)[..., None]
            preview_pixels = preview_pixels * alpha_array + (1 - alpha_array)
        previews.append((f"{image.scene} / {image.version}", np.clip(linear_to_srgb(preview_pixels), 0, 1)))
        contact_sources.append(
            {
                "source_master_sha256": image.source_sha256,
                "source_identity_sha256": image.source_identity_sha256,
                "preset": contact_preset,
                "derivative_sha256": contact_derivative_sha256,
                "render": preview_render,
            }
        )
        cached_display = display = None
        del master, data, preview_master
    contact = _write_new(root, "contact-sheet.jpg", _contact_sheet(previews))
    contact["sources"] = contact_sources
    contact["recipe"] = {
        "schema": "tp.photo.contact_sheet.v1",
        "max_columns": 4,
        "cell_size": [360, 290],
        "thumbnail_long_edge": 340,
        "jpeg_quality": 92,
        "icc_sha256": sha256(output_srgb_icc()),
        "render_selection": "last_requested_SDR_preset_or_explicit_perceptual_srgb_preview",
        "scope": "Bounded contact preview; native artifact inspection and human approval remain separate",
    }
    inventory: dict[str, Any] = {
        "schema": "tp.photo.delivery_inventory.v1",
        "property_name": property_name,
        "sources": sources,
        "files": records,
        "native_components": native_components,
        "contact_sheet": contact,
        "contact_sheet_composite": "white_background_for_transparency",
        "approval_scope": "Caller-approved source hashes only; derivatives do not inherit approval",
        "precision_policy": "Byte-preserved masters; float32 extended retouch; explicit SDR render for JPEG/print16. Container promotion never recovers source precision or clipped HDR values.",
        "quality_acceptance": "Human review required; integrity does not establish photographic or provider acceptance",
    }
    inventory_record = _write_new(root, "inventory.json", canonical_bytes(inventory))
    explanations: list[str] = [
        f"Photographic delivery: {property_name}",
        "",
        inventory["precision_policy"],
        inventory["quality_acceptance"],
        "Contact sheet: the last requested SDR render per source, or a perceptual-sRGB preview for master-only sets; thumbnails do not replace native inspection.",
        "",
        "File\tIntended use\tDimensions\tBits\tProfile\tSource master SHA-256",
    ]
    explanations.extend(
        f"{row['path']}\t{row['intended_use']}\t{row['width']}x{row['height']}\t{row['bit_depth']}\t{row['color_profile']}\t{row['source_master_sha256']}"
        for row in records
    )
    readme = _write_new(root, "inventory.txt", ("\n".join(explanations) + "\n").encode("utf-8"))
    selected = [*records, *native_components, contact, inventory_record, readme]
    spec = {
        "schema": "tp.delivery.package_spec.v1",
        "archive_root": f"{property_slug}-photographic-delivery",
        "files": [
            {"source": row["path"], "archive_path": row["path"], "sha256": row["sha256"], "size_bytes": row["size_bytes"]}
            for row in selected
        ],
    }
    _write_new(root, "package-spec.json", canonical_bytes(spec))
    return inventory


def package_delivery(output_dir: Path, archive: Path, *, copy_to: Path | None = None) -> dict[str, Any]:
    """Use the supported merged stdlib packager in source or installed layouts."""
    from transformation_portal.core import verified_delivery

    implementation = Path(verified_delivery.__file__)
    tool_digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
    try:
        receipt = verified_delivery.build_package(Path(output_dir) / "package-spec.json", output_dir, archive, copy_to)
    except verified_delivery.PackageError as error:
        raise ValueError(f"Verified packaging failed: {error}") from error
    if hashlib.sha256(implementation.read_bytes()).hexdigest() != tool_digest:
        raise ValueError("Verified packager source changed during packaging; generated bytes require a fresh verification")
    return {
        **receipt,
        "packager_source_sha256": tool_digest,
        "packager_implementation": "transformation_portal.core.verified_delivery",
    }
