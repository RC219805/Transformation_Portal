# V3+V2 Orchestrator Hardening Roadmap

**Status**: Implementation Guide
**Based On**: `docs/architecture/V3_ORCHESTRATOR_ARCHITECTURAL_REVIEW.md`
**Target**: Production-perfect orchestrator in 8-12 hours

---

## Overview

This roadmap provides concrete implementation steps for hardening the V3+V2 orchestrator from "production-ready" to "production-perfect" state. All critical security issues are already addressed; this focuses on operational correctness and user experience.

---

## Phase 1: Operational Correctness (Week 1 - 8 hours)

### Task 1.1: Collision-Free Output Paths (2 hours)

**Goal**: Prevent filename collisions when processing nested directory structures.

**Implementation**:

1. Add `make_output_key()` function to `orchestrator.py`:

```python
def make_output_key(input_path: Path, input_root: Path) -> Path:
    """Generate collision-free output key preserving directory structure.

    Args:
        input_path: Full path to input image
        input_root: Root directory of inputs

    Returns:
        Relative path suitable for output (without extension)

    Examples:
        renders/kitchen/view.jpg → kitchen/view
        renders/exterior/view.jpg → exterior/view
    """
    try:
        relpath = input_path.relative_to(input_root)
        return relpath.parent / relpath.stem
    except ValueError:
        # If input_path is not relative to input_root, use just the stem
        logger.warning(f"{input_path} is not relative to {input_root}, using flat naming")
        return Path(input_path.stem)
```

2. Update `EnhanceOrchestrator.__init__()` to track input root:

```python
class EnhanceOrchestrator:
    def __init__(self, config: EnhanceConfig, output_root: Path, input_root: Optional[Path] = None):
        self.config = config
        self.output_root = Path(output_root)
        self.input_root = Path(input_root) if input_root else None
        # ...
```

3. Update `enhance_image()` to use nested paths:

```python
def enhance_image(self, image_input: ImageInput) -> Dict[str, Any]:
    # Generate collision-free key
    if self.input_root:
        output_key = make_output_key(image_input.path, self.input_root)
    else:
        output_key = Path(image_input.path.stem)

    # Sanitize each component
    output_key = Path(*[sanitize_file_stem(p) for p in output_key.parts])

    # Build paths with nested structure
    depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
    combined_manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"
    # ...
```

4. Update `enhance_batch()` to pass input_root:

```python
def enhance_batch(self, input_dir: Path) -> List[Dict[str, Any]]:
    self.input_root = Path(input_dir)  # Track for collision prevention
    # ...
```

**Testing**:
```python
def test_collision_prevention(tmp_path):
    """Verify nested directories don't collide."""
    # Create test structure
    (tmp_path / "renders/kitchen").mkdir(parents=True)
    (tmp_path / "renders/exterior").mkdir(parents=True)

    # Create two view.jpg files
    create_test_image(tmp_path / "renders/kitchen/view.jpg")
    create_test_image(tmp_path / "renders/exterior/view.jpg")

    # Process
    orchestrator = EnhanceOrchestrator(config, tmp_path / "output", input_root=tmp_path / "renders")
    results = orchestrator.enhance_batch(tmp_path / "renders")

    # Verify no collisions
    assert (tmp_path / "output/depth/kitchen/view_depth.png").exists()
    assert (tmp_path / "output/depth/exterior/view_depth.png").exists()
```

---

### Task 1.2: Manifest-Based Resume (4 hours)

**Goal**: Prevent reuse of stale outputs when inputs or configuration change.

**Implementation**:

1. Add hash computation optimization:

```python
# manifest.py
def compute_file_sha256_cached(path: Path, cache: Optional[Dict[Path, str]] = None) -> str:
    """Compute SHA256 with optional caching to avoid recomputation."""
    if cache is not None and path in cache:
        return cache[path]

    hash_value = compute_file_sha256(path)

    if cache is not None:
        cache[path] = hash_value

    return hash_value
```

2. Add resume validation function:

```python
# orchestrator.py
def should_skip_depth(
    self,
    depth_path: Path,
    manifest_path: Path,
    image_input: ImageInput,
) -> bool:
    """Determine if depth generation can be safely skipped.

    Returns True only if:
    - Depth file exists
    - Combined manifest exists
    - Input image hash matches manifest
    - Model version matches manifest
    - Previous run succeeded
    """
    if not depth_path.exists():
        logger.debug("Depth file missing - will generate")
        return False

    if not manifest_path.exists():
        logger.warning(f"Depth exists but no manifest - regenerating for safety: {depth_path}")
        return False

    try:
        manifest = CombinedManifest.load(manifest_path)

        # Check input hash
        current_hash = compute_file_sha256(image_input.path)
        if manifest.input and manifest.input.image_sha256 != current_hash:
            logger.info(f"Input image changed - regenerating depth: {image_input.path}")
            return False

        # Check model version
        current_model = self.config.model_variant.value
        if manifest.depth and manifest.depth.model != current_model:
            logger.info(f"Model changed ({manifest.depth.model} → {current_model}) - regenerating")
            return False

        # Check quantization method
        current_quantization = self.config.depth_quantization
        if manifest.depth and manifest.depth.scaling.get("method") != current_quantization:
            logger.info(f"Quantization method changed - regenerating")
            return False

        # Check status
        if not manifest.depth or manifest.depth.get("status") != "ok":
            logger.warning("Previous depth run incomplete - regenerating")
            return False

        logger.debug(f"Resuming with existing depth: {depth_path}")
        return True

    except Exception as e:
        logger.warning(f"Manifest read failed: {e} - regenerating for safety")
        return False
```

3. Replace simple existence check:

```python
def enhance_image(self, image_input: ImageInput) -> Dict[str, Any]:
    # ...

    # Check resume conditions (OLD CODE)
    # skip_depth = depth_path.exists() and not self.config.force_depth

    # Check resume conditions (NEW CODE)
    skip_depth = (
        not self.config.force_depth
        and self.should_skip_depth(depth_path, combined_manifest_path, image_input)
    )

    # ...
```

**Testing**:
```python
def test_resume_detects_input_change(tmp_path, sample_image):
    """Verify resume detects when input image changes."""
    # First run
    orchestrator = EnhanceOrchestrator(config, tmp_path)
    result1 = orchestrator.enhance_image(ImageInput(sample_image))

    # Modify input image (change a pixel)
    img = Image.open(sample_image)
    pixels = img.load()
    pixels[0, 0] = (255, 0, 0)  # Change first pixel
    img.save(sample_image)

    # Second run: should regenerate depth
    orchestrator2 = EnhanceOrchestrator(config, tmp_path)
    with patch.object(orchestrator2.inference_engine, 'predict') as mock_predict:
        orchestrator2.enhance_image(ImageInput(sample_image))
        mock_predict.assert_called_once()  # Depth was regenerated
```

---

### Task 1.3: Enhanced Atomic Writes (2 hours)

**Goal**: Prevent partial file writes from corrupting outputs on crashes.

**Implementation**:

1. Add atomic write wrapper:

```python
# depth_writer.py
def atomic_write_depth_u16_png(
    path: Path,
    depth: np.ndarray,
    method: str = "p1p99",
    debug_verify: bool = False,
) -> Tuple[float, float]:
    """Write depth with atomic rename to prevent partial files.

    This ensures that if the process crashes during write, the output
    directory will not contain corrupt/partial files.

    Args:
        path: Final output path
        depth: Depth array to write
        method: Quantization method
        debug_verify: Enable verification

    Returns:
        Tuple of (p1, p99) percentile values
    """
    # Write to temporary file first
    tmp_path = path.with_suffix(".tmp.png")

    try:
        # Write depth to temp file
        p1, p99 = write_depth_u16_png(tmp_path, depth, method, debug_verify)

        # Atomic rename (POSIX guarantees atomicity)
        tmp_path.replace(path)

        logger.debug(f"Atomically wrote depth to {path}")
        return p1, p99

    except Exception:
        # Clean up partial write
        if tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.debug(f"Cleaned up partial write: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up {tmp_path}: {cleanup_error}")
        raise
```

2. Update orchestrator to use atomic writes:

```python
# orchestrator.py
def enhance_image(self, image_input: ImageInput) -> Dict[str, Any]:
    # ...

    if not skip_depth:
        try:
            depth_result = self.inference_engine.predict(image_input)
            depth_runtime_s = time.time() - start_time

            # Write depth atomically (CHANGED)
            p1, p99 = atomic_write_depth_u16_png(
                depth_path,
                depth_result.depth,
                method=self.config.depth_quantization,
                debug_verify=True,
            )

            # ...
```

3. Extend to manifests:

```python
# manifest.py
def atomic_write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Write JSON with atomic rename."""
    tmp_path = path.with_suffix(".tmp.json")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(data, indent=indent))
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

class CombinedManifest:
    def write(self, path: Path) -> None:
        """Write manifest atomically."""
        atomic_write_json(path, self.to_dict())
```

**Testing**:
```python
def test_atomic_write_cleans_up_on_failure(tmp_path):
    """Verify partial writes are cleaned up on failure."""
    output_path = tmp_path / "depth.png"

    # Simulate write failure
    with patch('PIL.Image.Image.save', side_effect=IOError("Disk full")):
        with pytest.raises(IOError):
            atomic_write_depth_u16_png(output_path, np.random.rand(100, 100).astype(np.float32))

    # Verify no partial file remains
    assert not output_path.exists()
    assert not (tmp_path / "depth.tmp.png").exists()
```

---

## Phase 2: Enhanced Provenance (Week 2 - 10 hours)

### Task 2.1: Enhance Manifest Schema (6 hours)

**Goal**: Add depth convention, toolchain versions, and clipping statistics.

**Implementation**:

1. Extend DepthMetadata dataclass:

```python
@dataclass
class DepthScalingMetadata:
    """Detailed depth quantization metadata."""
    method: str  # "p1p99", "p0.5p99.5", "minmax"
    p_low_percentile: float  # e.g., 1.0
    p_high_percentile: float  # e.g., 99.0
    v_low_value: float  # Actual depth value at p_low
    v_high_value: float  # Actual depth value at p_high
    clipped_low_frac: float  # Fraction of pixels clipped at low end
    clipped_high_frac: float  # Fraction of pixels clipped at high end
    invalid_frac: float  # Fraction of NaN/Inf pixels (pre-cleaning)

@dataclass
class DepthMetadata:
    backend: str
    model: str
    license: str
    non_commercial_ok: bool
    depth_path: str
    dtype: str
    shape: List[int]
    scaling: DepthScalingMetadata  # CHANGED: now a dataclass
    runtime_ms: float
    # NEW FIELDS
    representation: str = "depth"  # "depth" vs "inverse_depth" vs "disparity"
    convention: str = "higher_is_farther"  # vs "higher_is_nearer"
    invalid_policy: str = "nan_to_zero"  # How NaN/Inf were handled
    unit: str = "relative"  # "relative" vs "metric_meters"
```

2. Update quantization to compute clipping stats:

```python
# depth_writer.py
def write_depth_u16_png_with_stats(...) -> Tuple[float, float, DepthScalingMetadata]:
    """Write depth and return detailed scaling metadata."""
    # ... existing quantization code ...

    # Compute clipping statistics
    if method == "p1p99":
        p_low_percentile, p_high_percentile = 1.0, 99.0
    elif method == "p0.5p99.5":
        p_low_percentile, p_high_percentile = 0.5, 99.5
    elif method == "minmax":
        p_low_percentile, p_high_percentile = 0.0, 100.0

    clipped_low = (depth_f32 < p1).sum() / depth_f32.size
    clipped_high = (depth_f32 > p99).sum() / depth_f32.size
    invalid = (~np.isfinite(depth)).sum() / depth.size if depth.dtype != np.uint16 else 0.0

    scaling_meta = DepthScalingMetadata(
        method=method,
        p_low_percentile=p_low_percentile,
        p_high_percentile=p_high_percentile,
        v_low_value=float(p1),
        v_high_value=float(p99),
        clipped_low_frac=float(clipped_low),
        clipped_high_frac=float(clipped_high),
        invalid_frac=float(invalid),
    )

    return float(p1), float(p99), scaling_meta
```

3. Add toolchain version capture:

```python
@dataclass
class EnvironmentMetadata:
    """Toolchain and hardware environment."""
    python: str
    torch: Optional[str] = None
    cuda_runtime: Optional[str] = None
    gpu_name: Optional[str] = None
    driver: Optional[str] = None
    os: Optional[str] = None

def capture_environment() -> EnvironmentMetadata:
    """Capture current environment details."""
    import sys
    import platform

    env = EnvironmentMetadata(python=sys.version.split()[0], os=platform.system())

    try:
        import torch
        env.torch = torch.__version__
        if torch.cuda.is_available():
            env.cuda_runtime = torch.version.cuda
            env.gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return env

@dataclass
class CombinedManifest:
    schema: str = MANIFEST_SCHEMA_VERSION
    input: Optional[InputMetadata] = None
    depth: Optional[DepthMetadata] = None
    v2: Optional[V2Metadata] = None
    timing: Optional[TimingMetadata] = None
    repro: Optional[ReproMetadata] = None
    env: Optional[EnvironmentMetadata] = None  # NEW
```

---

### Task 2.2: Batch Summary Manifest (4 hours)

**Goal**: Generate batch-level JSON summarizing all images processed.

**Implementation**:

1. Add batch manifest dataclass:

```python
@dataclass
class BatchManifest:
    """Batch-level processing summary."""
    schema: str = "lux-depth-v3.batch.v1"
    batch_id: str
    start_time: str
    end_time: str
    config: Dict[str, Any]
    images: List[Dict[str, Any]]
    summary: Dict[str, Any]
```

2. Update `enhance_batch()` to generate summary:

```python
def enhance_batch(self, input_dir: Path) -> List[Dict[str, Any]]:
    """Process batch and generate summary manifest."""
    import datetime

    batch_id = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    start_time = datetime.datetime.now().isoformat()

    results = []
    for image_path in self._discover_images(input_dir):
        result = self.enhance_image(ImageInput(image_path))
        results.append(result)

    end_time = datetime.datetime.now().isoformat()

    # Build batch summary
    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "error": sum(1 for r in results if r.get("status") == "error"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "total_runtime_s": sum(r.get("timing", {}).get("total_s", 0) for r in results),
        "avg_runtime_s": sum(r.get("timing", {}).get("total_s", 0) for r in results) / len(results) if results else 0,
    }

    batch_manifest = BatchManifest(
        batch_id=batch_id,
        start_time=start_time,
        end_time=end_time,
        config=asdict(self.config),
        images=[
            {
                "stem": Path(r["image"]).stem,
                "status": r.get("status", "unknown"),
                "manifest": str(r.get("manifest_path", "")),
                "error": r.get("error") if r.get("status") == "error" else None,
            }
            for r in results
        ],
        summary=summary,
    )

    # Write batch manifest
    batch_manifest_path = self.manifests_dir / f"batch_{batch_id}.json"
    atomic_write_json(batch_manifest_path, asdict(batch_manifest))
    logger.info(f"Batch summary written to {batch_manifest_path}")

    return results
```

---

## Phase 3: User Experience (Week 3 - 4 hours)

### Task 3.1: CLI Convenience Options (3 hours)

Add production-friendly CLI flags:

```python
# cli.py enhance subcommand
@app.command()
def enhance(
    # ... existing args ...

    # NEW ARGS
    include: Optional[str] = typer.Option(
        None,
        "--include",
        help="Glob patterns to include (comma-separated, e.g., '*.jpg,*.png')",
    ),
    exclude: Optional[str] = typer.Option(
        None,
        "--exclude",
        help="Glob patterns to exclude (comma-separated, e.g., '*_mask.png')",
    ),
    max_images: Optional[int] = typer.Option(
        None,
        "--max-images",
        help="Maximum number of images to process (for testing)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print planned actions without executing",
    ),
    hash_mode: str = typer.Option(
        "if-manifest-exists",
        "--hash-mode",
        help="When to compute image hashes: always, if-manifest-exists, never",
    ),
):
    """Run V3 depth + V2 enhancement pipeline."""
    # ...
```

---

### Task 3.2: EXIF Orientation Normalization (1 hour)

Add explicit orientation handling:

```python
# input_manager.py or preprocessing.py
def normalize_exif_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and strip orientation tag."""
    from PIL import ImageOps

    # Apply orientation transformation
    image = ImageOps.exif_transpose(image)

    # Strip EXIF orientation to prevent double-application
    if hasattr(image, 'getexif'):
        exif = image.getexif()
        if exif and 0x0112 in exif:  # Orientation tag
            del exif[0x0112]
            logger.debug("Stripped EXIF orientation tag")

    return image
```

---

## Testing Strategy

### Unit Tests (Required)
- `test_make_output_key()` - collision prevention
- `test_should_skip_depth()` - resume logic
- `test_atomic_write_depth()` - crash recovery
- `test_batch_manifest()` - summary generation

### Integration Tests (Recommended)
- End-to-end V3→V2 with nested directories
- Resume after input change
- Resume after model version change
- Crash recovery (kill process mid-write)

---

## Success Criteria

**Phase 1 Complete When**:
- ✅ Nested directories process without collisions
- ✅ Resume detects stale inputs/models
- ✅ Crash recovery leaves no partial files

**Phase 2 Complete When**:
- ✅ Manifests include depth convention and toolchain info
- ✅ Batch summary JSON generated after run

**Phase 3 Complete When**:
- ✅ CLI supports --dry-run, --max-images, --include/exclude
- ✅ EXIF orientation handled consistently

---

## Deployment

After all phases complete:

1. Update README with new features
2. Add migration guide for existing users
3. Run full test suite
4. Deploy to staging environment
5. Process test batch (100+ images)
6. Review batch summary for anomalies
7. Deploy to production

**Estimated Total Time**: 8-12 hours across 2-3 weeks.
