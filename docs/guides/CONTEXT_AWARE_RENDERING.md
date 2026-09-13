# Architectural Context Extraction and Rendering Boundaries

The maintained PDF extractor is
[`scripts/analysis/architectural_context_extractor.py`](../../scripts/analysis/architectural_context_extractor.py).
The public compatibility entrypoint remains
[`scripts/architectural_context_extractor.py`](../../scripts/architectural_context_extractor.py).
It reads PDF text and embedded images, then writes a JSON context for review.

## Extract a PDF

From the repository root, using an interpreter with Pillow and PyMuPDF:

```bash
./scripts/setup/run_repo_python.sh scripts/architectural_context_extractor.py \
  "path/to/plans.pdf" \
  --output /tmp/tp-extracted-context \
  --verbose
```

For `plans.pdf`, the outputs are `/tmp/tp-extracted-context/plans_context.json`
and `/tmp/tp-extracted-context/plans_images/pageNN_imgNN.<embedded-format>`.
The image files retain their extracted format; they are not necessarily PNGs
or complete rendered PDF pages. Review these outputs before downstream use.

The extractor uses `page.get_text()`, keyword patterns, and a dimension regex.
It can suggest room names, materials, styles, and project fields when text
matches those patterns. It does not perform OCR, measure drawing geometry,
validate scale, resolve architectural units reliably, or certify dimensions.
Image-only plans may have no useful extracted text. Missing or incorrect
fields require review against the original documents; extraction success is
not proof that a room, finish, or measurement is correct.

The optional PyMuPDF import is checked when extraction begins. A successful
`--help` or import check does not prove PDF extraction. Check the supported
[setup guide](SETUP_GUIDE.md) before adding dependencies; this guide does not
introduce a new governed installation lane.

## Archived automatic rendering integration

The earlier automatic strategy and premium-rendering examples now live at:

- [`archive/scripts/context_aware_rendering.py`](../../archive/scripts/context_aware_rendering.py)
- [`archive/scripts/premium_context_pipeline.py`](../../archive/scripts/premium_context_pipeline.py)

Their room strategies, quality tiers, sample deliverables, and timing claims
are historical implementation context. They are not a maintained end-to-end
entrypoint or evidence of current rendering quality. In particular, the old
`scripts/context_aware_rendering.py` and `scripts/premium_context_pipeline.py`
commands no longer identify those files.

Other context helpers under `scripts/utilities/` and `scripts/pipelines/`
are separate integrations; their presence does not connect this extractor to
Lux execution automatically. For maintained image processing, use the
[Lux Depth V3 CLI guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md). A successful plan,
a successful inference run, and verified output artifacts are separate checks.

## Validation boundary

This documentation describes source behavior. It provides no universal
extraction speed, rendering throughput, quality score, or architectural
accuracy guarantee. Establish any such result with the actual documents,
runtime, model, configuration, and reviewed outputs.
