# Copilot Instructions — Transformation Portal

## Mission & Quality Bar
Transformation Portal is a production-grade image + video processing toolkit for luxury real estate rendering, architectural visualization, and editorial finishing.

Your output must prioritize:
- **Photorealism and architectural integrity** (no “AI mush,” no haloing, no over-sharpened edges).
- **Color science correctness** (HDR/SDR handling, metadata preservation, predictable transforms).
- **Client-safe deliverables** (stable filenames, deterministic presets, reproducible pipelines).
- **Performance and batch throughput** (no unnecessary model reloads; avoid per-file overhead).
- **CI reliability** (tests must run fast; heavy ML must be mocked/skipped by default).

Assume existing scripts are actively used in production. Avoid breaking changes unless explicitly requested.


## Repository Orientation (Know Where Things Live)
- `depth_pipeline/` — Depth Anything V2 integration (CoreML acceleration on Apple Silicon).
- `depth_pipeline/pipeline.py` — Main depth-aware processing pipeline.
- `depth_pipeline/processors/` — Depth-based image processors (zone-aware).
- `material_response.py` — Material Response core implementation.
- `lux_render_pipeline.py` — AI-powered render refinement (SDXL, ControlNet, Real-ESRGAN).
- `luxury_tiff_batch_processor.py` — 16-bit TIFF batch processor CLI.
- `luxury_video_master_grader.py` — Video grading + FFmpeg master output.
- `depth_tools.py` — Depth estimation utilities.
- `config/` — YAML presets (source of truth for pipeline behavior).
- `assets/luts/**` — .cube LUTs (film emulation, location aesthetics, material response).
- `tests/` — pytest suite (fast, deterministic, minimal external dependencies).
- `docs/` — Architecture, performance, and version history.

Notes:
- `src/transformation_portal/` is WIP and may be excluded from linting in CI; still write clean code there.
- Client deliverables directories (e.g., `09_Client_Deliverables/`) should not be treated as a place for core logic.


## Non-Negotiable Rules
1. **Do not load heavy ML models at import time.**
   - Lazy-load inside functions/classes.
   - Cache models after first load (process lifetime).
2. **Never use `shell=True` for FFmpeg calls.**
   - Build argument lists (`list[str]`) and call `subprocess.run(..., check=True)`.
3. **Preserve precision and metadata.**
   - TIFF: keep 16-bit when possible; do not silently downcast to 8-bit.
   - Preserve IPTC/XMP/GPS when supported by the workflow.
4. **Backwards compatibility is a feature.**
   - Avoid renaming CLI flags, preset names, or output naming conventions.
5. **Make the pipeline observable.**
   - Provide `--dry-run` for CLIs that build commands or long workflows.
   - Provide `--verbose` logging and keep normal output clean.


## Python Standards & Style
- Python: **3.10+**, tested on 3.10/3.11/3.12.
- Follow PEP 8; **max line length 127**.
- Prefer `pathlib.Path` over string paths.
- Prefer `dataclasses.dataclass` for config objects.
- Use type hints for public functions and non-trivial internal functions.
- Use `logging` (not `print`) for debug/info/warn paths; CLIs can print concise user-facing summaries.
- Keep functions single-purpose; separate concerns:
  - CLI parsing
  - business logic
  - I/O + subprocess calls

### Errors & Optional Dependencies
- Optional features must fail gracefully with actionable errors:
  - If `tifffile` missing, explain how to install extras (`.[tiff]`).
  - If `torch/diffusers/transformers` missing, explain `.[ml]`.
- Use clear exception types and messages; avoid swallowing exceptions.


## Patterns to Follow (Project-Local Conventions)

### CLI (Typer)
- Use Typer consistently for new/updated CLIs.
- Provide:
  - `--preset` (config-driven workflow)
  - `--dry-run` (print commands/plan, do not execute)
  - `--verbose` (more logs)
- Validate inputs early (existence, file extensions, readable files) before any heavy work.
- Output directory behavior:
  - Support timestamped outputs when appropriate.
  - Follow `{basename}_{preset}.{ext}` unless explicitly changed.

### FFmpeg (Video and LUT Application)
- Build filter graphs with a dedicated function (e.g., `build_filter_graph()`).
- Preserve color metadata: `color_primaries`, `color_trc`, `colorspace` when possible.
- For HDR:
  - Detect PQ/HLG with ffprobe metadata.
  - Apply tone mapping with configurable operators (Hable/Reinhard/Mobius).
  - Prefer deterministic, testable logic that maps metadata → processing decisions.
- Always support a dry-run mode that prints the final FFmpeg command.

Implementation guidelines:
- Use `shutil.which("ffmpeg")` / `shutil.which("ffprobe")` checks.
- Use `subprocess.run(args, check=True, capture_output=..., text=True)` as appropriate.
- Never string-concatenate commands.

### Depth Pipeline (Depth Anything V2 + CoreML)
- Maintain ordering: **depth estimation → zone assignment → enhancements → tone mapping → finishing**.
- Use caching:
  - Depth map cache should be keyed by stable identifiers (e.g., absolute path + mtime + preset hash).
  - Prefer LRU caching for iterative workflows.
- Apple Silicon:
  - Prefer CoreML/ANE path when available; otherwise fallback to Torch CPU/MPS.
  - Detection must be robust and should not crash on non-macOS machines.
- Tests must not require downloading models.
  - Use stubs/mocks for depth model outputs (synthetic depth maps).

### Material Response Technology
- Treat Material Response as a **finishing** layer unless explicitly instructed otherwise.
- Avoid “style drift”: enhancements should respect highlights/midtones and preserve specular behavior.
- For surface detection:
  - Keep it modular and testable (pure functions where possible).
  - Make strength parameters explicit and bounded.

### LUT Handling
- LUT paths are relative to repo and must exist at runtime.
- Default LUT strength should be configurable; typical working range is ~0.6–0.8.
- Stacking LUTs must be deterministic (order matters). Document stacking order in code and docs.
- Video LUT application should use FFmpeg; image LUT application can use the project’s Python implementation (when available).


## Testing Requirements (pytest)
- Add/modify tests with any functional change.
- Keep unit tests fast and deterministic:
  - Mock FFmpeg execution and file I/O where possible.
  - Mock ML model loading/inference.
- For optional dependencies:
  - Use `pytest.importorskip()` or markers to skip cleanly.
- Prefer fixtures for shared setup; avoid duplicating temporary directory logic.
- Add edge cases:
  - missing file / unreadable file
  - invalid preset
  - HDR metadata present/absent
  - unusual frame rates (23.976, 29.97, VFR detection)
  - 16-bit TIFF round-trip precision (when tifffile available)

When a bug is fixed, add a regression test that would have failed before the fix.


## Performance & Resource Discipline
- Avoid per-file model initialization.
- Avoid converting large images multiple times; minimize copies.
- Use batch processing for I/O-bound operations.
- Use multiprocessing cautiously; avoid oversubscribing CPU/GPU:
  - Default to safe concurrency.
  - Provide a `--workers` option where it materially helps.
- If you introduce caching, make cache invalidation explicit and testable.


## Documentation & Versioning
When behavior changes:
- Update the relevant docs in `docs/`.
- Update `docs/Version_History/changelog.md` for significant changes.
- If you add/modify presets:
  - Document intended use case and rationale in YAML and/or preset notes.
  - Add at least one targeted test.

README must reflect current CLI options and examples.


## Security & Safety Hygiene
- Treat input files and paths as untrusted.
- Validate paths; avoid path traversal when mirroring directory trees.
- Do not embed credentials, tokens, or private endpoints.
- Do not add network calls that run by default (downloads must be opt-in).


## “Good Output” Checklist (Use Before Finalizing Code)
- No heavy imports executed at module import time.
- FFmpeg invoked safely (args list, no shell).
- Metadata preservation considered.
- Tests added/updated and avoid external heavyweight dependencies.
- CLI remains backward compatible.
- Logs are informative; dry-run supported where relevant.
- Docs/changelog updated when behavior changes.