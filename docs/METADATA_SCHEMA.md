# Metadata Schema Documentation

## Overview

This document defines the schema versioning strategy for manifest metadata classes in the Transformation Portal lux_depth_v3 pipeline. Schema versioning prevents silent metadata corruption by explicitly tracking data structure versions and enforcing compatibility rules.

## Schema Version: 1.0

**Status**: Current
**Release**: 2025-01-30
**Stability**: Stable

### InputMetadata Schema 1.0

The `InputMetadata` class captures metadata about input images processed by the pipeline.

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `schema_version` | `str` | No | `"1.0"` | Schema version identifier for compatibility checking |
| `image_path` | `str` | Yes | - | Path to input image (relative or absolute) |
| `image_sha256` | `str` | No | `None` | SHA256 hash of image file for integrity verification |
| `image_size_bytes` | `int` | No | `None` | Size of image file in bytes |
| `image_dimensions` | `tuple[int, int]` | No | `None` | Image dimensions as (width, height) |

#### Example

```python
from src.transformation_portal.lux_depth_v3.manifest import InputMetadata

# Create metadata
metadata = InputMetadata(
    image_path="/path/to/image.jpg",
    image_sha256="abc123def456...",
    image_size_bytes=1024000,
    image_dimensions=(1920, 1080),
    schema_version="1.0",  # Optional, defaults to "1.0"
)

# Serialize to dictionary
data = metadata.to_dict()

# Deserialize from dictionary (with schema validation)
restored = InputMetadata.from_dict(data)
```

#### Serialization Format (JSON)

```json
{
  "schema_version": "1.0",
  "image_path": "/path/to/image.jpg",
  "image_sha256": "abc123def456...",
  "image_size_bytes": 1024000,
  "image_dimensions": [1920, 1080]
}
```

**Note**: `image_dimensions` is serialized as a list in JSON but automatically converted to a tuple during deserialization.

---

## Schema Evolution Rules

### Adding New Fields

When adding a new field to `InputMetadata`:

1. **Make it optional** with a sensible default value
2. **Increment the schema version** (e.g., `1.0` → `1.1` for backward-compatible changes)
3. **Update `from_dict()`** to handle missing fields gracefully
4. **Update this documentation** with the new field specification
5. **Add migration tests** to verify old manifests can be loaded

**Example: Adding a new optional field**

```python
@dataclass
class InputMetadata:
    # Existing fields...
    schema_version: str = "1.1"  # Incremented

    # New optional field
    exif_metadata: Optional[Dict[str, Any]] = None  # NEW
```

### Breaking Changes

When making a breaking change (e.g., renaming a field, changing field type):

1. **Increment the major version** (e.g., `1.0` → `2.0`)
2. **Implement a migration function** to upgrade old manifests
3. **Update `from_dict()`** to detect old schema versions and migrate automatically
4. **Document the migration path** in this file
5. **Deprecate the old schema version** with a clear timeline

**Example: Migration pattern**

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> InputMetadata:
    schema_version = data.get('schema_version', '1.0')

    if schema_version == '1.0':
        # Migrate 1.0 -> 2.0
        data = _migrate_1_0_to_2_0(data)
        schema_version = '2.0'

    if schema_version != '2.0':
        raise ValueError(f"Unsupported schema version: {schema_version}")

    return cls(**data)
```

### Removing Fields

**Policy**: Fields should never be removed. Instead:

1. **Deprecate the field** by marking it as `Optional[...]` with default `None`
2. **Document the deprecation** in this file
3. **Remove usage** in code but keep field definition for compatibility
4. **After 6 months**, consider removing in a major version bump with migration

### Compatibility Guarantees

#### Backward Compatibility

Code supporting schema version `N` **must** be able to read manifests from schema version `N-1` (or earlier) by:
- Providing sensible defaults for missing fields
- Migrating deprecated fields to new equivalents
- Raising clear errors for truly incompatible data

#### Forward Compatibility

Code supporting schema version `N` **must** reject manifests from schema version `N+1` (or later) by:
- Detecting unsupported `schema_version` values
- Raising a descriptive `ValueError` with guidance on upgrading

**Example Error Message**:
```
ValueError: Unsupported InputMetadata schema version: 2.0.
This code supports version 1.0 only. Please upgrade to the latest version.
```

---

## Testing Requirements

All schema changes **must** include:

1. **Roundtrip tests**: Verify serialize → deserialize → equality
2. **Migration tests**: Verify old manifests can be loaded
3. **Forward compatibility tests**: Verify newer schemas are rejected
4. **Edge case tests**: Empty values, unicode, very long strings, etc.

See `tests/test_metadata_roundtrip.py` for comprehensive test coverage.

---

## Schema History

### Version 1.0 (2025-01-30)

**Initial Release**

- Fields: `schema_version`, `image_path`, `image_sha256`, `image_size_bytes`, `image_dimensions`
- Rationale: Fix silent corruption bug from positional argument construction (Issue #758)
- Enforcement: Roundtrip stability tests prevent regression

**Changes from Pre-Versioning**:
- Added explicit `schema_version` field (defaults to `"1.0"`)
- Added `to_dict()` and `from_dict()` methods for safe serialization
- Added schema validation in `from_dict()` to reject unsupported versions
- Enforced keyword-only construction pattern in documentation

---

## Related Documentation

- **Roundtrip Tests**: `tests/test_metadata_roundtrip.py`
- **Manifest Implementation**: `src/transformation_portal/lux_depth_v3/manifest.py`
- **Issue #758**: InputMetadata positional args bug (root cause)
- **PR #764**: Atomic write pattern unification (prevention)

---

## FAQ

### Why schema versioning?

Schema versioning prevents silent corruption by:
1. Making data structure contracts explicit
2. Enabling detection of incompatible changes
3. Supporting gradual migration during upgrades
4. Providing clear error messages when versions mismatch

### What happens if I forget to increment the schema version?

The roundtrip stability tests in `test_metadata_roundtrip.py` will catch breaking changes:
- If you change field types, tests will fail
- If you rename fields, tests will fail
- If you change serialization format, tests will fail

### Can I skip schema versioning for internal-only changes?

**No.** All changes to serialized data structures must follow schema evolution rules:
- Manifests are persisted to disk and may be read by future versions
- CI/CD pipelines may use manifests from different code versions
- Users may archive manifests for long-term reproducibility

### How do I debug schema version mismatches?

1. Check the `schema_version` field in the JSON manifest
2. Compare with the supported version in `InputMetadata.from_dict()`
3. Look for migration functions or deprecation warnings
4. Consult the Schema History section above

---

## Enforcement

Schema versioning is enforced by:

1. **Type system**: Fields are typed with `Optional[...]` for clarity
2. **Tests**: Roundtrip tests in `test_metadata_roundtrip.py` (664 tests as of 2025-01-30)
3. **Runtime validation**: `from_dict()` validates `schema_version` before deserialization
4. **Code review**: All manifest changes require Architect approval
5. **Documentation**: This file serves as the normative schema reference

---

## Contact

For questions or schema change proposals, consult:
- **Architect**: Schema evolution policy and breaking changes
- **Specialist**: Implementation details and test coverage
- **Issue Tracker**: GitHub Issues for bugs and feature requests
