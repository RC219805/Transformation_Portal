# APEX Tier 1: Registry Public API Migration

**Status**: ✅ Complete
**PR**: #879
**Date**: 2026-02-09

## Summary

Tier 1 refactors dependency validation to use a clean public API on `DepthBackendRegistry`, eliminating direct access to internal `._backends` and enforcing fail-fast behavior for unknown backends.

## Changes

### Registry Public API (New)

Added three public methods to `DepthBackendRegistry`:

```python
def get_backend_class(backend_id: str) -> Optional[Type[DepthBackend]]:
    """Get backend class by ID without instantiation."""

def available_backend_ids() -> list[str]:
    """Get sorted list of all registered backend IDs."""

def has_backend(backend_id: str) -> bool:
    """Check if backend is registered."""
```

### Unknown Backend Handling (Breaking Change)

**Before**: Unknown `backend_id` silently fell back to strict dependency check (torch + transformers).

**After**: Unknown `backend_id` raises `ValueError` with clear guidance:

```
Unknown backend_id 'typo'.
Available backends: da3, depth_pro, mock

Fix: choose a valid backend_id or register the backend.
See: docs/apex/phase3/README.md for backend registration.
```

## Benefits

1. **Encapsulation**: No external code touches `._backends` internals
2. **Fail-fast**: Configuration errors caught immediately
3. **Clear errors**: Users see exactly what went wrong
4. **Extensibility**: Adding new API methods doesn't break code

## Testing

All 8 backend dependency tests pass. Key coverage:
- ✅ Unknown backends fail fast with clear message
- ✅ DA3 requires torch + transformers
- ✅ Non-HF backends don't require transformers
- ✅ Broken imports treated as missing

## Related

- **Next**: Tier 2 (skip dependency checks in dry-run mode)
- **See**: Issue #875, docs/apex/phase3/README.md
