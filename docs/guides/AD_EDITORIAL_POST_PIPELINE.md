# Editorial RAW post-production workflow

The maintained entrypoint is `tools/ad_editorial_post_pipeline.py`. It preserves
RAW offload/backup, selects, style variants, optional HDR and panoramas, TIFF/JPEG
export, contact sheets, metadata, and the existing project layout. The code is
an editorial finishing tool; the name does not establish Architectural Digest
approval or production photographic acceptance.

## Runtime and setup

The supported full runtime is isolated from core and ML environments. Its
checked-in closure targets **native Apple Silicon, macOS 14+, CPython 3.12**.
Use a trusted Python 3.12 builder; the repository `.venv/bin/python` is suitable
only when it is that version. Run from the repository root:

```bash
.venv/bin/python -I scripts/setup/editorial_runtime.py install
.venv/bin/python -I scripts/setup/editorial_runtime.py check
```

`make install-editorial-runtime` and `make check-editorial-runtime` are aliases
for those commands, using the selected repository interpreter.

The installer consumes [the target-owned editorial lock](../../requirements/locks/README.md),
including rawpy, ReportLab, ExifRead, and piexif. Shared NumPy, Pillow, SciPy,
OpenCV, tifffile, imagecodecs, PyYAML, and tqdm versions remain constrained by the
governed core lock. No existing core/ML lock or environment is modified. The
separate RAW worker installer does not supply the full editorial closure.

Installation downloads only recorded PyPI wheel URLs, verifies SHA256 hashes,
and installs offline with exact pins. It stages a new generation under
`.runtime/editorial/generations/`, checks dependency consistency and real image/PDF
encoding, then atomically switches `.runtime/editorial/current`. Existing and
failed generations are retained; an unsuccessful install preserves the current
runtime. Use an owned path without symlinked ancestors or group/other write access.
For a custom location, put `--runtime-root /absolute/private/editorial-runtime`
before `install`, `check`, or `run` in every command.

Every governed check/run authenticates the retained wheels and installed package
payload before execution, rejects startup hooks and dependency/contract drift,
and ignores old bytecode caches. Keep the runtime and repository under the same
trusted user; this is not a sandbox against concurrent modifications by that user.
Changing the Python builder requires reinstalling a fresh generation. Do not
install additional packages or edit runtime contents in place.

The default smoke proves RGB uint16 TIFF with independent OpenCV decoding,
JPEG with an embedded ICC profile, and a PDF containing an image. It does not
prove RAW camera compatibility or photographic acceptance. Add an owned fixture
for real RAW decoding:

```bash
.venv/bin/python -I scripts/setup/editorial_runtime.py \
  --raw-file /absolute/private/scene.CR3 check
```

`check --contract-only` validates committed input/lock/core-pin identity without
installing anything and works without the native runtime. It does not execute
encoders. ExifTool remains an optional system binary for IPTC/XMP; the governed
piexif fallback writes only the documented JPEG EXIF fields.

## Configure and run

```bash
cp tools/sample_config.yml /absolute/private/editorial.yml
```

Set `project_root` to a dedicated output project and `input_raw_dir` to the
photographic inputs. Review backup and rename settings before running: these
copy originals, can rename the project copies, and can reuse an existing project.
Use a fresh project for comparisons. Configure `selects` deliberately; CSV paths
and metadata paths resolve relative to the project root unless absolute. An
all-rejected selects CSV selects no images and the run fails without exports.
RAW discovery includes NEF, RAF, ORF, CR2/CR3, DNG, ARW, RW2, SRW, and CRW.
Offload flattens source folders and rejects duplicate basenames before copying;
selected output stems must also be unique. Create the configured metadata CSV
before running, or set `metadata: {}` to use an optional default CSV. Style names must be single directory
names, so they cannot redirect outputs outside the project layout.

```bash
.venv/bin/python -I scripts/setup/editorial_runtime.py run -- \
  --config /absolute/private/editorial.yml -vv
```

The CLI and directory contract remain:

```text
Project/
├── RAW/Originals/
├── WORK/BaseTIFF/
├── WORK/Aligned/
├── WORK/Variants/<style>/
├── EXPORT/Print_TIFF/<style>/
├── EXPORT/Web_JPEG/<style>/
└── DOCS/
    ├── selects.csv
    ├── metadata.csv
    ├── ContactSheets/
    └── Manifests/manifest.json
```

TIFF suffixes remain `.tif`; web images remain `.jpg`. Image writes retain atomic
replacement through uniquely created sibling temporary files. A failed encoding preserves an
existing destination. Successful per-image work can remain after another image
fails; any recorded processing failure or a run without exports returns a failed
CLI status. Inspect the exit status and manifest together. The complete project,
metadata operations, manifest, and optional ZIP are not one atomic transaction.

## Color and numeric contract

RAW decoding uses camera white balance, AHD, disabled auto-brightness, 16-bit
samples, and linear ProPhoto RGB. Editing uses float32 RGB in a bounded `[0,1]`
working range. This is not an unclipped scene-referred archival master.

TIFFs contain genuine three-channel uint16 samples. With `icc: {}`, intermediate
and print TIFFs use a deterministic linear ProPhoto matrix/shaper profile. When
`icc.prophoto_path` is supplied, it must have matching ProPhoto primaries and
supported identical monotone channel transfer curves; samples are encoded to
that profile's transfer before saving and decoded back when reopened. Profile
bytes are preserved. Missing, mismatched, LUT-based, malformed, and unsupported
profiles fail validation. A profile description alone is not color authority.

Web export resizes and sharpens the linear master, transforms ProPhoto/D50 through
Bradford adaptation to linear sRGB/D65, then applies the sRGB transfer function.
Only then does it quantize to 8-bit JPEG and embed an sRGB profile. Omitted
`icc.srgb_path` uses a deterministic built-in sRGB profile; supplied profiles must
match sRGB primaries and transfer. Values outside destination sRGB are clipped;
there is no perceptual gamut-mapping claim. The colorimetric matrices follow the
[W3C color conversion reference](https://www.w3.org/TR/css-color-4/#color-conversion-code).

Saturation, split tone, resize, sharpening, and dust reconstruction use floating
samples. Eight-bit proxies are limited to geometric/spot detection. TIFF storage
still introduces uint16 quantization at each intermediate write; optional edits
and clipping can reduce detail. Correct output encoding is not photographic
quality acceptance.

## HDR brackets

```yaml
processing:
  enable_hdr: true
  hdr_group_gap_sec: 2.0
```

Grouping uses consecutive name-sorted groups of three to five captures and
whole-second EXIF timestamps within the configured adjacent gap. Missing
timestamps cannot create a fictitious bracket. HDR admission requires distinct positive measured shutter
times plus unchanged aperture and ISO. Inputs must have identical geometry and
already be aligned; camera motion and moving subjects require external review.
Use the same camera, sensor mode, and fixed white balance throughout a bracket;
those capture invariants are operator preconditions, not automatically verified.

Because RAW input is already linear, the implementation merges saturation-weighted
radiance using the actual shutter times, rather than estimating a response curve
from 8-bit images with invented equal exposures. It uses the median exposure as
a reference and a shared logarithmic shoulder normalized by the 99.9th percentile
peak, then clips to the unit-range delivery contract. The legacy Python helper
name `hdr_merge_debvec` remains for compatibility. The existing `enable_hdr` flag
selects the repaired linear-input implementation. This changes HDR pixels and
requires fresh paired photographic acceptance; it is not a radiometrically
calibrated HDR deliverable or an automatic deghosting/alignment system.

## Panorama groups

```yaml
processing:
  enable_pano: true
  pano_groups:
    - ["IMG_0001.CR3", "IMG_0002.CR3", "IMG_0003.CR3"]
```

Names must match selected, post-rename RAW copies. Supply adjacent overlapping
views in order. SIFT features and robust homographies are estimated from temporary
8-bit sRGB proxies. The original linear ProPhoto float images are warped and
feather-blended; the displayed proxy is never the panorama master. Insufficient
matches, invalid geometry, missing inputs, or an excessive canvas fail the run.

This is a planar homography model, without lens calibration, spherical projection,
parallax correction, exposure matching, seam optimization, or content-aware fill.
A rectangular output may retain black borders. Review architecture, overlap
seams, and perspective before using a panorama as a deliverable. Real multi-view
acceptance remains separate from synthetic geometry and single-photo crop tests.

## Grading, metadata, and performance limits

Styles retain exposure, contrast, saturation, split tone, and cinematic vignette
controls. Contrast uses a monotone luminance curve around 18% linear gray,
preserving black/white and channel ratios before range clipping. It no longer
subtracts a contrast offset around linear 0.5, which crushed shadows.
Luminance-based operations use the ProPhoto luminance coefficients.
Split-tone hue interpolation follows the shortest circular path. Auto-upright
performs a small global rotation from detected lines; it is not full perspective
correction. Dust removal is an optional local inpainting heuristic, not semantic
retouching.

Metadata is read from CSV; an explicitly configured CSV must exist. When no path
is configured, the default `<project>/DOCS/metadata.csv` is optional. ExifTool supplies the IPTC/XMP path; it operates on a
sibling temporary copy, and a failed write fails the run while preserving the
prior image. The `piexif` fallback writes JPEG Artist/Copyright EXIF fields only,
merging existing EXIF without re-encoding pixels or removing the ICC profile.
Missing fallback dependencies fail an attempted JPEG metadata write. TIFF
IPTC/XMP requires ExifTool; without it, the tool warns and adds no TIFF metadata.
Verify actual output tags; these paths do not preserve every camera metadata
field from RAW conversion or certify an editorial delivery schema. Contact sheets
use exported web JPEGs and fail on an unreadable image, preserving any prior PDF
through atomic replacement. Serial execution is current behavior: `processing.workers`
is retained and validated but does not enable parallel processing. HDR, panoramas,
and per-style consistency hold full image arrays in memory. No universal runtime,
throughput, or memory guarantee is established.

## Verification

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_ad_editorial_post_pipeline_paths.py -q
```

Focused checks cover real RGB TIFF sample/tag round trips, independent OpenCV
reading, supplied transfer encoding, a LittleCMS reference color transform,
atomic failure preservation, float grading precision, measured-exposure HDR, and
bounded float panorama composition. Local RAW runs and rendered comparisons are
additional evidence; they do not replace reviewed capture brackets, real panorama
sets, dependency closure, platform coverage, or operator acceptance.

Historical snapshots remain under `tools/deprecated/` and
`docs/historical/script-audits/`. They do not define current runtime or color
contracts. Repository licensing is defined in the root [LICENSE](../../LICENSE).
