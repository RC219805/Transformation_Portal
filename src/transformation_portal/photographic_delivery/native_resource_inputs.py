"""Closed, allocation-bounded TIFF eligibility before photographic decoding.

This is resource evidence, not photographic approval or decoding authority.
Unknown containers and metadata retain the conservative resource profile. No
untrusted TIFF, ICC, XML, EXIF, or pixel payload reaches an opaque parser here.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from typing import Any

from transformation_portal.ingest.canonical_json import dumps_json

NATIVE_RASTER_SCHEMA = "tp.photo.native_raster_evidence.v1"
NATIVE_RASTER_PROFILE = "tiff_srgb_integer_contiguous_v1"
MAX_SOURCE_BYTES = 512 * 1024**2
MAX_PIXELS = 30_000_000
MAX_EDGE = 65536
MAX_HEADER_BYTES = 64 * 1024
MAX_TAGS = 32
MAX_STRIPS = 64
MAX_DESCRIPTION_BYTES = 1024
MAX_ICC_BYTES = 4096

# Only baseline scalar/strip fields, bounded descriptive metadata and sRGB ICC.
# Unknown tags include tiled geometry, SubIFDs, EXIF, XMP and other authorities.
_TAG_TYPES = {
    256: (3, 4, 16),
    257: (3, 4, 16),
    258: (3,),
    259: (3,),
    262: (3,),
    270: (2,),
    273: (3, 4, 16),
    274: (3,),
    277: (3,),
    278: (3, 4, 16),
    279: (3, 4, 16),
    282: (5,),
    283: (5,),
    284: (3,),
    296: (3,),
    305: (2,),
    338: (3,),
    339: (3,),
    34675: (7,),
}
_TYPE_BYTES = {2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 16: 8}
_INTEGER_FORMAT = {3: "H", 4: "I", 16: "Q"}


def _canonical(value: Any) -> bytes:
    return dumps_json(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


@dataclass(frozen=True, slots=True)
class NativeRasterEvidence:
    """Scalar source/layout bindings; retained bytes must be revalidated on use."""

    source_sha256: str
    size_bytes: int
    width: int
    height: int
    source_width: int
    source_height: int
    alpha: bool
    source_bit_depth: int
    orientation: int
    byte_order: str
    container: str
    header_bytes: int
    tag_count: int
    strip_count: int
    raster_bytes: int
    icc_sha256: str
    profile_sha256: str
    layout_sha256: str
    header_sha256: str

    def to_payload(self) -> dict[str, Any]:
        """Return bounded, detached evidence without source bytes or paths."""
        return {
            "schema": NATIVE_RASTER_SCHEMA,
            "profile": NATIVE_RASTER_PROFILE,
            **asdict(self),
            "resource_limits": {
                "source_bytes": MAX_SOURCE_BYTES,
                "pixels": MAX_PIXELS,
                "edge": MAX_EDGE,
                "header_bytes": MAX_HEADER_BYTES,
                "tags": MAX_TAGS,
                "strips": MAX_STRIPS,
                "description_bytes": MAX_DESCRIPTION_BYTES,
                "icc_bytes": MAX_ICC_BYTES,
            },
        }

    def content_hash(self) -> str:
        """Hash the descriptor without introducing a self-referential field."""
        return hashlib.sha256(_canonical(self.to_payload())).hexdigest()


@dataclass(frozen=True)
class _Tag:
    kind: int
    count: int
    data: memoryview


class _Ineligible(ValueError):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Ineligible


def _tag_limit(code: int) -> int:
    if code in {273, 279}:
        return MAX_STRIPS
    if code == 270:
        return MAX_DESCRIPTION_BYTES
    if code == 305:
        return 64
    if code == 34675:
        return MAX_ICC_BYTES
    return 4 if code in {258, 339} else 1


def _directory(raw: bytes) -> tuple[str, str, dict[int, _Tag], int]:
    view = memoryview(raw)
    _require(len(raw) >= 8 and raw[:2] in {b"II", b"MM"})
    order = "<" if raw[:2] == b"II" else ">"
    version = struct.unpack_from(order + "H", view, 2)[0]
    if version == 42:
        prefix, count_bytes, entry_bytes, value_bytes, offset_format = 8, 2, 12, 4, "I"
        offset = struct.unpack_from(order + "I", view, 4)[0]
        count_format, container = "H", "classic_tiff"
    else:
        _require(version == 43 and len(raw) >= 16)
        _require(struct.unpack_from(order + "HH", view, 4) == (8, 0))
        prefix, count_bytes, entry_bytes, value_bytes, offset_format = 16, 8, 20, 8, "Q"
        offset = struct.unpack_from(order + "Q", view, 8)[0]
        count_format, container = "Q", "bigtiff"
    _require(prefix <= offset <= MAX_HEADER_BYTES - count_bytes and offset + count_bytes <= len(raw))
    count = struct.unpack_from(order + count_format, view, offset)[0]
    _require(1 <= count <= MAX_TAGS)
    end = offset + count_bytes + count * entry_bytes + value_bytes
    _require(end <= min(len(raw), MAX_HEADER_BYTES))
    _require(struct.unpack_from(order + offset_format, view, end - value_bytes)[0] == 0)
    intervals = [(0, prefix), (offset, end)]
    tags: dict[int, _Tag] = {}
    previous = -1
    header_end = end
    for index in range(count):
        start = offset + count_bytes + index * entry_bytes
        code, kind = struct.unpack_from(order + "HH", view, start)
        number = struct.unpack_from(order + offset_format, view, start + 4)[0]
        _require(code > previous and code in _TAG_TYPES and kind in _TAG_TYPES[code])
        _require(kind != 16 or version == 43)
        _require(1 <= number <= _tag_limit(code))
        previous = code
        size = number * _TYPE_BYTES[kind]
        value_start = start + 4 + value_bytes
        if size > value_bytes:
            value_start = struct.unpack_from(order + offset_format, view, value_start)[0]
            value_end = value_start + size
            _require(prefix <= value_start < value_end <= min(len(raw), MAX_HEADER_BYTES))
            _require(all(value_end <= first or value_start >= last for first, last in intervals))
            intervals.append((value_start, value_end))
            header_end = max(header_end, value_end)
        tags[code] = _Tag(kind, number, view[value_start : value_start + size])
    return order, container, tags, header_end


def _integers(tags: dict[int, _Tag], code: int, order: str, default: tuple[int, ...] = ()) -> tuple[int, ...]:
    if code not in tags:
        return default
    tag = tags[code]
    _require(tag.kind in _INTEGER_FORMAT)
    return struct.unpack(order + _INTEGER_FORMAT[tag.kind] * tag.count, tag.data)


def _scalar(tags: dict[int, _Tag], code: int, order: str, default: int = 0) -> int:
    values = _integers(tags, code, order, (default,))
    _require(len(values) == 1)
    return values[0]


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _description(tag: _Tag, shape: list[int], bits: int) -> None:
    data = bytes(tag.data)
    _require(data.endswith(b"\0") and b"\0" not in data[:-1])
    text = data[:-1].decode("ascii")
    # This lexical ceiling runs before JSON parsing: one object, one shape list,
    # bounded keys/strings and no escapes, nested objects, floats or constants.
    _require(re.fullmatch(r'[A-Za-z0-9_ .,:{}\[\]"\r\n\t-]+', text) is not None)
    _require(text.count("{") == text.count("}") == 1 and text.count("[") == text.count("]") == 1)
    _require(text.count(":") <= 5 and text.count('"') <= 24)
    _require(all(len(number) <= 8 for number in re.findall(r"[0-9]+", text)))
    payload = json.loads(text, object_pairs_hook=_unique)
    expected = {
        "shape": shape,
        "color_space": "srgb",
        "color_domain": "display_encoded_srgb",
        "image_artifact_schema": "tp.image.master.v1",
        "precision": "uint8" if bits == 8 else "uint16",
    }
    _require(isinstance(payload, dict) and "shape" in payload and set(payload) <= set(expected))
    _require(all(payload[key] == expected[key] for key in payload))
    _require(all(type(value) is int for value in payload["shape"]))


def _inspect(raw: bytes, source_sha256: str) -> NativeRasterEvidence:
    order, container, tags, header_end = _directory(raw)
    width, height = _scalar(tags, 256, order), _scalar(tags, 257, order)
    _require(0 < width <= MAX_EDGE and 0 < height <= MAX_EDGE and width * height <= MAX_PIXELS)
    channels = _scalar(tags, 277, order)
    _require(channels in {3, 4} and _scalar(tags, 262, order) == 2)
    bits_values = _integers(tags, 258, order)
    _require(len(bits_values) == channels and len(set(bits_values)) == 1 and bits_values[0] in {8, 16})
    bits = bits_values[0]
    _require(_scalar(tags, 259, order) == 1 and _scalar(tags, 284, order, 1) == 1)
    _require(_integers(tags, 339, order, (1,) * channels) == (1,) * channels)
    _require(_integers(tags, 338, order) == ((2,) if channels == 4 else ()))
    orientation = _scalar(tags, 274, order, 1)
    _require(orientation in range(1, 9))
    rows = _scalar(tags, 278, order)
    _require(1 <= rows <= height)
    offsets, counts = _integers(tags, 273, order), _integers(tags, 279, order)
    strips = (height + rows - 1) // rows
    _require(len(offsets) == len(counts) == strips <= MAX_STRIPS)
    _require(header_end <= offsets[0] <= MAX_HEADER_BYTES)
    end = offsets[0]
    for index, (offset, count) in enumerate(zip(offsets, counts)):
        expected = min(rows, height - index * rows) * width * channels * (bits // 8)
        _require(offset == end and count == expected and offset + count <= len(raw))
        end = offset + count
    _require(end == len(raw))
    if 270 in tags:
        _description(tags[270], [height, width, channels], bits)
    if 305 in tags:
        software = bytes(tags[305].data)
        # Vendor Software values can select additional tifffile metadata paths.
        # The qualified producer is deliberately narrower than TIFF acceptance.
        _require(software == b"tifffile.py\0")
    for code in (282, 283):
        if code in tags:
            numerator, denominator = struct.unpack(order + "II", tags[code].data)
            _require(0 < numerator <= 1_000_000 and 0 < denominator <= 1_000_000)
    _require(_scalar(tags, 296, order, 1) in {1, 2, 3})
    _require(34675 in tags)
    profile = bytes(tags[34675].data)
    _require(132 <= len(profile) <= MAX_ICC_BYTES)
    # Generate only a trusted fixed profile; never parse the supplied ICC in CMS.
    from transformation_portal.lux_depth_v4.color_preparation import normalize_icc, output_srgb_icc

    canonical_profile = output_srgb_icc()
    _require(normalize_icc(profile) == canonical_profile)
    layout = {
        "container": container,
        "byte_order": order,
        "shape": [height, width, channels],
        "bits": bits,
        "orientation": orientation,
        "rows_per_strip": rows,
        "offsets": offsets,
        "counts": counts,
    }
    return NativeRasterEvidence(
        source_sha256=source_sha256,
        size_bytes=len(raw),
        width=width if orientation < 5 else height,
        height=height if orientation < 5 else width,
        source_width=width,
        source_height=height,
        alpha=channels == 4,
        source_bit_depth=bits,
        orientation=orientation,
        byte_order="little" if order == "<" else "big",
        container=container,
        header_bytes=offsets[0],
        tag_count=len(tags),
        strip_count=strips,
        raster_bytes=sum(counts),
        icc_sha256=hashlib.sha256(profile).hexdigest(),
        profile_sha256=hashlib.sha256(canonical_profile).hexdigest(),
        layout_sha256=hashlib.sha256(_canonical(layout)).hexdigest(),
        header_sha256=hashlib.sha256(memoryview(raw)[: offsets[0]]).hexdigest(),
    )


def inspect_native_raster(raw: bytes, expected_sha256: str) -> NativeRasterEvidence | None:
    """Classify exact immutable bytes; all unsupported headers keep fallback cost.

    Type or source-identity mismatches are errors. Eligibility is not a promise
    that decoding, managed admission, runtime qualification or human review pass.
    """
    if type(raw) is not bytes or type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Native raster inspection requires immutable bytes and a lowercase SHA-256")
    if not 0 < len(raw) <= MAX_SOURCE_BYTES:
        return None
    if hashlib.sha256(memoryview(raw)).hexdigest() != expected_sha256:
        raise ValueError("Native raster source SHA-256 changed")
    try:
        return _inspect(raw, expected_sha256)
    except (ValueError, TypeError, struct.error, IndexError):
        return None
