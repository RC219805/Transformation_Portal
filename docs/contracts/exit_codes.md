# Exit Code Contract (Phase 4 Tooling)

## Status

- Normative for Phase 4 capture/provenance tooling.
- Companion to ADR-035.

## Interpretation Rule

Exit-code routing MUST use `(tool, exit_code)` as the semantic key.

Numeric values are intentionally reused across tools with different meanings, so
consumers MUST NOT interpret a bare integer without tool context.

## Reserved Ranges

- `0`: Success
- `1`: Unhandled/runtime failure (not part of deterministic contract surfaces)
- `2-9`: Phase 4 tool-local deterministic failures
- `30-39`: Cross-runtime determinism/parity gate failures

## Current Assignments

### `tools/extract_capture_metadata.py` (`tp.meta.capture.v1` extraction)

- `0`: success
- `2`: config invalid
- `3`: path normalization failure
- `4`: extraction failure
- `5`: schema validation failure
- `6`: strict-mode warning failure

### `tools/build_metadata_manifest.py` (`tp.meta.capture_manifest.v1` build)

- `0`: success
- `2`: input read/parse error
- `3`: input invariant failure
- `4`: schema validation failure
- `5`: manifest write failure

### `tools/build_provenance_manifest.py` (`tp.meta.provenance.v1` build)

- `0`: success
- `2`: input read/parse error
- `3`: input invariant failure
- `4`: schema validation failure
- `5`: manifest write failure

### `tools/build_provenance_merkle.py` (`tp.meta.provenance_merkle.v1` build)

- `0`: success
- `2`: input read/parse error
- `3`: input invariant failure
- `4`: schema validation failure
- `5`: merkle write failure

### Cross-runtime parity gates

- `tools/check_phase4d_manifest_cross_runtime.py`
  - `0`: success
  - `31`: runtime invocation failure
  - `32`: parity mismatch
- `tools/check_bundle_root_cross_runtime.py`
  - `0`: success
  - `31`: runtime invocation failure
  - `32`: parity/root mismatch
- `tools/check_governance_export_cross_runtime.py`
  - `0`: success
  - `31`: runtime invocation failure
  - `32`: parity mismatch

## Change Control

Changing the meaning of an assigned `(tool, exit_code)` pair is a contract
change and requires ADR update plus tests.
