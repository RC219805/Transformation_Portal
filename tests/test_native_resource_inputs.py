"""Only closed, bounded raster layouts may select native resource qualification."""

from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import tifffile

from transformation_portal.lux_depth_v4.color_preparation import output_srgb_icc
from transformation_portal.lux_depth_v4.photography import decode_master
from transformation_portal.photographic_delivery import DeliveryImage
from transformation_portal.photographic_delivery import native_resource_inputs as inputs
from transformation_portal.photographic_delivery.materials import MaterialsRecipe, create_materials_revision

pytestmark = pytest.mark.unit


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _raster(*, bits=16, alpha=False, orientation=1, byteorder="<", bigtiff=False, shape=(7, 9), **options):
    channels = 4 if alpha else 3
    dtype = np.dtype("uint8" if bits == 8 else "uint16")
    samples = np.arange(shape[0] * shape[1] * channels, dtype=dtype).reshape(*shape, channels)
    samples[..., :3] //= 2
    if alpha:
        samples[..., -1] = np.iinfo(dtype).max
    buffer = io.BytesIO()
    kwargs = {
        "photometric": "rgb",
        "extrasamples": "unassalpha" if alpha else None,
        "iccprofile": output_srgb_icc(),
        "metadata": {"color_space": "srgb", "color_domain": "display_encoded_srgb"},
        "extratags": [(274, "H", 1, orientation, False)],
        "byteorder": byteorder,
        "bigtiff": bigtiff,
    }
    kwargs.update(options)
    tifffile.imwrite(buffer, samples, **kwargs)
    return buffer.getvalue()


def _directory(raw):
    order = "<" if raw[:2] == b"II" else ">"
    big = struct.unpack_from(order + "H", raw, 2)[0] == 43
    offset = struct.unpack_from(order + ("Q" if big else "I"), raw, 8 if big else 4)[0]
    count = struct.unpack_from(order + ("Q" if big else "H"), raw, offset)[0]
    count_bytes, entry_bytes, value_bytes = (8, 20, 8) if big else (2, 12, 4)
    return order, offset, count, count_bytes, entry_bytes, value_bytes


def _entry(raw, code):
    order, offset, count, count_bytes, entry_bytes, value_bytes = _directory(raw)
    for index in range(count):
        start = offset + count_bytes + index * entry_bytes
        if struct.unpack_from(order + "H", raw, start)[0] == code:
            return start, value_bytes
    raise AssertionError(f"Missing fixture tag {code}")


def _values(raw, code, replacement):
    raw = bytearray(raw)
    order = _directory(raw)[0]
    start, value_bytes = _entry(raw, code)
    kind = struct.unpack_from(order + "H", raw, start + 2)[0]
    count = struct.unpack_from(order + ("Q" if value_bytes == 8 else "I"), raw, start + 4)[0]
    fmt = {3: "H", 4: "I", 16: "Q"}[kind]
    location = start + 4 + value_bytes
    if struct.calcsize(fmt) * count > value_bytes:
        location = struct.unpack_from(order + ("Q" if value_bytes == 8 else "I"), raw, location)[0]
    old = list(struct.unpack_from(order + fmt * count, raw, location))
    values = replacement(old) if callable(replacement) else replacement
    struct.pack_into(order + fmt * count, raw, location, *values)
    return bytes(raw)


def _inspect(raw):
    return inputs.inspect_native_raster(raw, _sha(raw))


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("bigtiff", [False, True])
@pytest.mark.parametrize("orientation", [1, 6])
def test_native_resource_evidence_matches_supported_roundtrip(bits, alpha, byteorder, bigtiff, orientation):
    raw = _raster(bits=bits, alpha=alpha, byteorder=byteorder, bigtiff=bigtiff, orientation=orientation)
    evidence = _inspect(raw)
    master = decode_master(raw, source_name="source.tif")
    assert evidence is not None
    assert (evidence.height, evidence.width) == master.shape
    assert evidence.alpha == alpha and evidence.source_bit_depth == bits
    assert evidence.source_sha256 == _sha(raw) and evidence.size_bytes == len(raw)
    assert evidence.raster_bytes == 7 * 9 * (4 if alpha else 3) * (bits // 8)
    assert evidence.byte_order == ("little" if byteorder == "<" else "big")
    assert evidence.container == ("bigtiff" if bigtiff else "classic_tiff")
    assert evidence.header_bytes <= inputs.MAX_HEADER_BYTES and evidence.strip_count == 1
    assert evidence == _inspect(raw)
    assert len(json.dumps(evidence.to_payload())) < 2048


def test_native_resource_evidence_is_frozen_detached_and_source_bound():
    raw = _raster()
    evidence = _inspect(raw)
    with pytest.raises(FrozenInstanceError):
        evidence.width = 1
    assert not hasattr(evidence, "__dict__")
    before = evidence.content_hash()
    payload = evidence.to_payload()
    payload["width"] = 1
    payload["resource_limits"]["pixels"] = 1
    assert evidence.content_hash() == before
    altered = raw[:-1] + bytes([raw[-1] ^ 1])
    other = _inspect(altered)
    assert other is not None and other.source_sha256 != evidence.source_sha256
    assert other.layout_sha256 == evidence.layout_sha256 and other.content_hash() != before


def test_native_resource_inspection_does_not_call_opaque_container_or_pixel_decoders(monkeypatch):
    raw = _raster()

    def forbidden(*args, **kwargs):
        raise AssertionError("Eligibility invoked an opaque decoder")

    monkeypatch.setattr(tifffile, "TiffFile", forbidden)
    monkeypatch.setattr("PIL.Image.open", forbidden)
    monkeypatch.setattr("transformation_portal.lux_depth_v4.photography.decode_master", forbidden)
    assert _inspect(raw) is not None


def test_native_resource_source_hash_consumes_a_buffer_without_copying(monkeypatch):
    raw = _raster()
    expected = _sha(raw)
    original = inputs.hashlib.sha256
    full_source_buffers = []

    def track(value=b""):
        if len(value) == len(raw):
            full_source_buffers.append(value)
        return original(value)

    monkeypatch.setattr(inputs.hashlib, "sha256", track)
    assert inputs.inspect_native_raster(raw, expected) is not None
    assert len(full_source_buffers) == 1
    assert isinstance(full_source_buffers[0], memoryview) and full_source_buffers[0].obj is raw


@pytest.mark.parametrize("orientation", range(1, 9))
def test_native_resource_orientation_preserves_native_axis_mapping(orientation):
    evidence = _inspect(_raster(orientation=orientation))
    assert evidence is not None
    assert (evidence.width, evidence.height) == ((9, 7) if orientation < 5 else (7, 9))
    assert (evidence.source_width, evidence.source_height) == (9, 7)


def test_native_resource_accepts_retained_delivery_metadata_and_generated_materials_tiff():
    description = json.dumps(
        {
            "shape": [7, 9, 3],
            "color_space": "srgb",
            "image_artifact_schema": "tp.image.master.v1",
            "color_domain": "display_encoded_srgb",
        }
    )
    raw = _raster(description=description, metadata=None)
    assert _inspect(raw) is not None
    recipe = MaterialsRecipe.from_payload(
        {
            "schema": "tp.photo.materials_recipe.v1",
            "input_stage": "display_referred",
            "regions": [{"id": "glass", "label": "glass", "region": [0, 0, 1, 1], "strength": 0}],
        }
    )
    result = create_materials_revision(DeliveryImage("source.tif", _sha(raw), "room", source_bytes=raw), recipe)
    assert _inspect(result["source_bytes"]) is not None


@pytest.mark.parametrize("raw", [b"", b"II", b"II*\0", b"II+\0", b"MM\0*\0\0\0\xff", b"not-a-tiff", b"\x89PNG\r\n\x1a\n"])
def test_native_resource_tiny_or_unsupported_headers_keep_fallback(raw):
    assert _inspect(raw) is None


@pytest.mark.parametrize("bigtiff", [False, True])
@pytest.mark.parametrize(
    "damage", ["offset", "count", "cycle", "tag_type", "tag_count", "tag_overlap", "header_offset", "duplicate"]
)
def test_native_resource_rejects_malformed_directory_without_following_offsets(bigtiff, damage):
    raw = bytearray(_raster(bigtiff=bigtiff))
    order, offset, count, count_bytes, entry_bytes, value_bytes = _directory(raw)
    integer = "Q" if bigtiff else "I"
    if damage == "offset":
        struct.pack_into(order + integer, raw, 8 if bigtiff else 4, len(raw) + 100)
    elif damage == "count":
        struct.pack_into(order + ("Q" if bigtiff else "H"), raw, offset, inputs.MAX_TAGS + 1)
    elif damage == "cycle":
        struct.pack_into(order + integer, raw, offset + count_bytes + count * entry_bytes, offset)
    elif damage in {"tag_type", "tag_count", "tag_overlap", "header_offset"}:
        start, _ = _entry(raw, 34675)
        if damage == "tag_type":
            struct.pack_into(order + "H", raw, start + 2, 99)
        elif damage == "tag_count":
            struct.pack_into(order + integer, raw, start + 4, 2**32 - 1)
        else:
            value = offset if damage == "tag_overlap" else inputs.MAX_HEADER_BYTES + 1
            struct.pack_into(order + integer, raw, start + 4 + value_bytes, value)
    else:
        first = struct.unpack_from(order + "H", raw, offset + count_bytes)[0]
        struct.pack_into(order + "H", raw, offset + count_bytes + entry_bytes, first)
    assert _inspect(bytes(raw)) is None


@pytest.mark.parametrize("damage", ["overlap", "gap", "count", "trailing", "truncated"])
def test_native_resource_strips_must_exactly_cover_contiguous_native_geometry(damage):
    raw = _raster(rowsperstrip=3)
    assert _inspect(raw).strip_count == 3
    if damage in {"overlap", "gap"}:
        delta = -1 if damage == "overlap" else 1
        raw = _values(raw, 273, lambda values: [values[0], values[1] + delta, values[2]])
    elif damage == "count":
        raw = _values(raw, 279, lambda values: [values[0] + 1, *values[1:]])
    elif damage == "trailing":
        raw += b"extra"
    else:
        raw = raw[:-1]
    assert _inspect(raw) is None


def test_native_resource_rejects_padded_tiles_and_excessive_segments():
    assert _inspect(_raster(tile=(16, 16))) is None
    assert _inspect(_raster(shape=(inputs.MAX_STRIPS, 2), rowsperstrip=1)).strip_count == inputs.MAX_STRIPS
    assert _inspect(_raster(shape=(inputs.MAX_STRIPS + 1, 2), rowsperstrip=1)) is None


@pytest.mark.parametrize(
    "description",
    [
        '{"shape":[7,9,3],"unknown":1}',
        '{"shape":[7,9,3],"color_space":"linear_srgb"}',
        '{"shape":[7,9,3],"color_domain":"extended_encoded_srgb"}',
        '{"shape":[7,9,3],"shape":[7,9,3]}',
        '{"shape":[7.0,9,3]}',
        '{"shape":[true,9,3]}',
        '{"shape":[7,9,3],"precision":"float32"}',
        '{"shape":[7,9,3],"color_space":"s\\u0072gb"}',
        '{"shape":[7,9,3]} trailing',
    ],
)
def test_native_resource_description_is_a_closed_bounded_contract(description):
    assert _inspect(_raster(description=description, metadata=None)) is None


@pytest.mark.parametrize("description", ['{"shape":' + "[" * 400 + "0" + "]" * 400 + "}", "x" * 1025])
def test_native_resource_description_expansion_is_rejected_before_json_parse(description, monkeypatch):
    raw = _raster(description=description, metadata=None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unbounded description reached JSON parsing")

    monkeypatch.setattr(inputs.json, "loads", forbidden)
    assert _inspect(raw) is None


def test_native_resource_profile_must_match_recognized_srgb_bytes():
    profile = bytearray(output_srgb_icc())
    profile[24:36] = bytes(12)
    profile[84:100] = b"x" * 16
    assert _inspect(_raster(iccprofile=bytes(profile))) is not None
    profile[40] ^= 1
    assert _inspect(_raster(iccprofile=bytes(profile))) is None
    assert _inspect(_raster(iccprofile=None)) is None
    assert _inspect(_raster(iccprofile=b"x" * (inputs.MAX_ICC_BYTES + 1))) is None


@pytest.mark.parametrize("code,values", [(259, [5]), (262, [1]), (274, [9]), (284, [2]), (258, [8, 16, 16])])
def test_native_resource_rejects_unsupported_sample_semantics(code, values):
    assert _inspect(_values(_raster(), code, values)) is None


def test_native_resource_rejects_associated_alpha_unknown_tags_and_subifds():
    assert _inspect(_values(_raster(alpha=True), 338, [1])) is None
    assert _inspect(_raster(extratags=[(65000, "I", 1, 1, False)])) is None
    raw = bytearray(_raster())
    start, _ = _entry(raw, 305)
    struct.pack_into("<HHII", raw, start, 330, 4, 1, 8)
    assert _inspect(bytes(raw)) is None


@pytest.mark.parametrize("software", ["ScanImage", "Adobe Photoshop", "x" * 65])
def test_native_resource_vendor_metadata_keeps_conservative_fallback(software):
    assert _inspect(_raster(software=software)) is None


def test_native_resource_source_size_and_geometry_ceilings_keep_fallback(monkeypatch):
    raw = _raster()
    monkeypatch.setattr(inputs, "MAX_PIXELS", 62)
    assert _inspect(raw) is None
    monkeypatch.setattr(inputs, "MAX_PIXELS", 30_000_000)
    monkeypatch.setattr(inputs, "MAX_SOURCE_BYTES", len(raw) - 1)
    assert _inspect(raw) is None


@pytest.mark.parametrize("raw,expected", [(bytearray(b"x"), "a" * 64), (b"x", "bad"), (b"x", None), (b"x", "A" * 64)])
def test_native_resource_identity_inputs_are_not_coerced(raw, expected):
    with pytest.raises(ValueError, match="immutable bytes"):
        inputs.inspect_native_raster(raw, expected)


def test_native_resource_source_hash_mismatch_is_an_error():
    with pytest.raises(ValueError, match="SHA-256 changed"):
        inputs.inspect_native_raster(_raster(), "a" * 64)
