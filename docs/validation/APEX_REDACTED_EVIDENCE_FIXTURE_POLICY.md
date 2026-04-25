# APEX Redacted Evidence Fixture Policy

## Purpose

This policy defines the narrow conditions under which APEX evidence fixtures may
be committed for schema regression, parser stability, or renderer stability
tests.

Committed APEX evidence fixtures are allowed only for schema regression and
parser/renderer stability. They are never real APEX quality evidence and must
never satisfy promotion eligibility.

## Default Rule

Real generated APEX evidence, canonical references, RAW files, TIFF/TIF masters,
ICC/profile binaries, Materials V3 candidate outputs, private image assets,
delivery images, and generated `output/` bundles must remain outside git.

Real canonical evidence remains external-asset-only and requires non-synthetic
runs against mounted canonical references.

## Allowed Fixtures

Committed fixtures may be used only when all of these are true:

- The fixture is tiny enough for repository policy.
- The fixture is synthetic or fully redacted.
- The fixture exists only for schema regression, parser stability, or renderer
  stability.
- The fixture is labeled non-promotional.
- The fixture cannot satisfy APEX promotion eligibility.

## Prohibited Fixtures

Do not commit:

- real canonical 16-bit TIFF/TIF references
- RAW/DNG/CR2/CR3/NEF/ARW/RAF/ORF/RW2 files
- ICC/profile binaries
- real Materials V3 candidate outputs
- real generated evidence bundles
- private property imagery
- large JPEG/PNG delivery images
- delivery images that could reconstruct the underlying property asset
- generated `output/` artifacts from real APEX runs

## Non-Promotional Requirement

Any committed fixture must use `synthetic_data=true` or equivalent
non-promotional labeling.

For evidence-shaped fixtures, promotion must remain blocked. Prefer
`promotion_verdict: blocked` and a `promotion_blocked_reasons` value such as
`synthetic_data` or an equivalent `fixture_non_promotional` reason when the
fixture shape supports those fields.

```json
{
  "run": {
    "synthetic_data": true
  },
  "promotion_verdict": "blocked",
  "promotion_blocked_reasons": [
    "synthetic_data"
  ]
}
```

Raw CLIP similarity, synthetic metrics, redacted images, or fixture-only
telemetry must never authorize Materials V3 pixel operations or real APEX
promotion.

## Future Fixture PR Checklist

Any future PR adding a committed APEX fixture must document:

- fixture purpose
- fixture size
- fixture provenance
- synthetic or redaction method
- why docs-only coverage is insufficient
- why the fixture cannot expose private real-estate imagery
- confirmation that it cannot satisfy promotion eligibility
- confirmation that promotion eligibility remains blocked

If any checklist item cannot be satisfied, keep the artifact outside git and use
the external asset-root workflow instead.
