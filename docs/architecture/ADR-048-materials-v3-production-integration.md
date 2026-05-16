# ADR-048: Materials V3 Production Integration

**Status:** Approved
**Date:** 2026-02-11
**Decider:** Transformation Portal Architect
**Context:** Materials V3 stub implementation complete, need production-ready integration

> **2026-05-16 renumbering note:** Originally filed as `ADR-030`, which collided with the canonical `ADR-030: Phase II Deterministic RAW Ingest` (see `ADR-030-phase2-deterministic-raw-ingest.md`). Renumbered to `ADR-048` (next free slot) to resolve the collision. No content changes.

---

## Context

Materials V3 infrastructure exists but is currently a stub:

### Current State
1. **Segmentation Backend:**
   - Default: `material_segmentation_backend="stub"` (returns empty masks)
   - Alternative: `material_segmentation_backend="efficientsam"` (real ML segmentation)
   - Segmentation disabled by default: `enable_material_segmentation=False`

2. **V2 Integration:**
   - `v2_enhance.py` accepts `material_masks` parameter (prepared, not wired)
   - Orchestrator computes masks but doesn't pass them to V2 subprocess
   - V2 runner doesn't support passing masks via CLI

3. **Gap:**
   - No mechanism to serialize masks to disk for subprocess consumption
   - No CLI argument in `scripts/enhance_image.py` to accept mask paths
   - No preset examples showing production configuration

### Problem Statement

To make Materials V3 production-ready:
1. Enable real segmentation in recommended configurations
2. Wire material masks through orchestrator → V2 subprocess boundary
3. Maintain backward compatibility with existing workflows
4. Ensure cleanup of temporary serialization artifacts

---

## Decision

Implement two-phase integration:

### Phase 1: Enable Real Segmentation (Item 1)

**Configuration Strategy:**
- Keep `enable_material_segmentation=False` as safe default
- Add opt-in presets demonstrating production configuration
- Ensure graceful fallback: EfficientSAM unavailable → stub backend with warning

**Implementation:**
- No code changes needed (infrastructure exists)
- Create example preset: `config/materials_v3_production.yaml`
- Document recommended settings and performance implications

### Phase 2: Wire Masks to V2 Subprocess (Item 2)

**Serialization Strategy:**
```
output_root/
  temp/
    {output_key.stem}_materials_v3_masks.npz  # Compressed mask bundle
```

**Format:**
- Use NumPy `.npz` (compressed archive)
- Keys: material names (e.g., "glass", "water", "stone")
- Values: float32 masks [0.0, 1.0], shape (H, W)

**CLI Extension:**
```bash
python scripts/enhance_image.py input.jpg \
  --depth-dir depth/ \
  --output-dir output/ \
  --masks-dir temp/  # NEW: optional masks directory
```

**V2 Runner Extension:**
```python
v2_runner.run(
    input_path=...,
    depth_dir=...,
    output_dir=...,
    masks_dir=...,  # NEW: optional masks directory
)
```

**Lifecycle:**
1. **Create:** Orchestrator serializes masks after Materials V3 processing
2. **Consume:** V2 subprocess loads masks from `--masks-dir`
3. **Cleanup:** Orchestrator deletes temporary masks after V2 completes

---

## Architectural Constraints

### Backward Compatibility
- Existing workflows must continue to work (no masks = valid)
- V2 subprocess must handle missing `--masks-dir` gracefully
- Default behavior unchanged (`enable_material_segmentation=False`)

### Security Constraints
- **Path Sanitization:** Mask file paths must be validated (prevent traversal)
- **Cleanup Guarantee:** Temporary masks deleted even on V2 failure
- **Size Limits:** Reject unreasonably large mask files (>100MB)

### Performance Constraints
- Serialization overhead: <100ms per image (target)
- Compression: Use `.npz` (typically 10-20% of uncompressed size)
- No memory leaks: Masks released after serialization

### Quality Firewall
- No regression in existing test suite
- V2 subprocess must handle masks=None without errors
- Mask serialization failures must not break pipeline (fall back to no masks)

---

## Implementation Plan

### 1. Mask Serialization (orchestrator.py)

Add helper function:
```python
def _serialize_material_masks(
    masks: Dict[str, np.ndarray],
    output_key: Path,
    temp_dir: Path,
) -> Optional[Path]:
    """Serialize material masks to NPZ file.

    Returns:
        Path to .npz file or None on failure
    """
```

Call after Materials V3 processing:
```python
# After materials_v3_result is available
masks_path = None
if materials_v3_result and materials_v3_result.get("material_masks"):
    masks_path = _serialize_material_masks(
        materials_v3_result["material_masks"],
        output_key,
        temp_dir,
    )
```

### 2. V2 Runner Extension (v2_runner.py)

Add parameter:
```python
def run(
    self,
    ...
    masks_file: Optional[Path] = None,  # NEW: Explicit NPZ file path
    **kwargs,
) -> Dict[str, Any]:
```

Pass to CLI:
```python
if masks_file is not None:
    cmd.extend(["--masks-file", str(masks_file)])
```

### 3. CLI Extension (scripts/enhance_image.py)

Add argument:
```python
parser.add_argument(
    "--masks-file",
    type=Path,
    default=None,
    help="Explicit path to material masks NPZ file (Materials V3 integration)",
)
```

Load and pass to `enhance_image()`:
```python
masks = load_material_masks(args.masks_file) if args.masks_file else None
result = enhance_image(
    input_path=...,
    masks_file=args.masks_file,  # Pass explicit file path
    ...
)
```

### 4. Cleanup (orchestrator.py)

Add cleanup in try-finally:
```python
try:
    # V2 subprocess execution
    v2_result = self.v2_runner.run(...)
finally:
    # Clean up temporary masks
    if masks_path and masks_path.exists():
        masks_path.unlink()
        logger.debug(f"Cleaned up temporary masks: {masks_path}")
```

---

## Testing Strategy

### Unit Tests
1. **Mask Serialization:**
   - Round-trip: serialize → deserialize → verify equality
   - Empty masks dict → returns None
   - Invalid masks (wrong dtype, shape) → handled gracefully

2. **V2 Runner:**
   - With masks_dir → CLI includes `--masks-dir`
   - Without masks_dir → CLI omits flag
   - Backward compatibility: existing calls work unchanged

3. **CLI:**
   - `--masks-dir` present → loads masks
   - `--masks-dir` missing → masks=None
   - Invalid mask file → error with clear message

### Integration Tests
1. **End-to-End Flow:**
   - Orchestrator with Materials V3 enabled
   - Verify masks serialized, passed to V2, cleaned up
   - Verify V2 subprocess receives and uses masks

2. **Backward Compatibility:**
   - Existing presets work unchanged
   - No masks → V2 subprocess runs normally

3. **Failure Scenarios:**
   - V2 subprocess fails → masks still cleaned up
   - Mask serialization fails → pipeline continues (no masks)
   - Mask file missing in V2 → graceful fallback

---

## Consequences

### Positive
- Materials V3 becomes production-ready
- Material-aware tone mapping enabled in V2 subprocess
- Backward compatible (existing workflows unaffected)
- Proper cleanup (no leaked temporary files)

### Negative
- Additional I/O overhead (serialize/deserialize masks)
- Temporary disk space required (~5-20MB per image)
- More complex orchestration logic

### Mitigations
- Use compressed `.npz` format (minimize disk usage)
- Cleanup in finally block (guarantee no leaks)
- Make masks optional everywhere (graceful degradation)

---

## Alternatives Considered

### Alternative 1: In-Memory Masks (Shared Memory)
**Rejected:** Requires complex IPC, fragile across Python versions

### Alternative 2: JSON Serialization
**Rejected:** Inefficient for large binary masks (10x larger than NPZ)

### Alternative 3: Embed Masks in Enhanced Image (Metadata)
**Rejected:** PNG doesn't support arbitrary binary metadata, TIFF metadata bloat

---

## References

- **Governance:** `docs/architecture/agent_governance.md`
- **V2 Enhancement:** `docs/architecture/V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md`
- **Materials V3 Taxonomy:** `src/transformation_portal/lux_depth_v3/materials_v3_taxonomy.py`
- **Segmentation Backend:** `src/transformation_portal/lux_depth_v3/segmentation_backend.py`

---

## Approval

**Architect Decision:** Approved
**Rationale:** Design is backward compatible, properly scoped, and enforceable via tests.
**Next Steps:** Proceed with implementation per plan above.
