# Archive Manifest Phase 3.1 - Detached Signing Architecture

## Status
Proposed (Phase 3.1 - Attestation Layer)

## Context

Phase 3 establishes a deterministic integrity contract:
- Deterministic per-file SHA-256 digests
- Deterministic Merkle aggregation
- Deterministic verification reports
- Byte-for-byte golden CI enforcement

Phase 3 artifacts MUST remain deterministic and offline.

Phase 3.1 introduces optional cryptographic attestation
without modifying Phase 3 artifacts or altering their bytes.

---

# Design Principles

1. Phase 3 artifacts are immutable inputs.
2. Signing is detached.
3. No Phase 3 artifact is rewritten or mutated.
4. No network calls occur.
5. No nondeterminism is introduced.
6. Signing never runs automatically in CI.
7. Root-only signing is prohibited.

---

# What Is Signed

Phase 3.1 signs the exact bytes of:

```text
merkle_roots.json
```

Rationale:

- Merkle roots encode the full integrity state.
- File digests are already committed within Merkle structure.
- The JSON artifact includes:
  - hash_algorithm
  - leaf_hash_algorithm
  - leaf_format_version
  - tree_method_version
  - partition roots
  - global root
  - leaf counts

Signing the entire JSON preserves:
- method metadata
- algorithm metadata
- leaf counts
- ordering semantics

Signing root-only is explicitly disallowed.

---

# Canonical Byte Definition

The signature input MUST be the exact bytes of `merkle_roots.json`
as written by Phase 3:

- No reserialization
- No whitespace normalization
- No field reordering
- No canonicalization transforms

---

# Signature Format

Detached JSON envelope:

```text
merkle_roots.sig.json
```

Example:

```json
{
  "signature_algorithm": "ed25519",
  "signed_artifact": "merkle_roots.json",
  "signed_artifact_sha256": "<hex>",
  "signature_base64": "<base64>"
}
```

The signature covers the raw bytes of the artifact.

---

# Deterministic Envelope Serialization

`merkle_roots.sig.json` MUST be serialized with:
- UTF-8 encoding
- sorted keys
- two-space indentation
- separators fixed to `(",", ": ")`
- trailing newline

This locks formatting behavior across Python versions and keeps emitted
envelopes byte-stable for deterministic workflows.

---

# CLI Interface

## Sign

```bash
python tools/sign_merkle_roots.py \
  --roots merkle_roots.json \
  --private-key private_key.pem \
  --out merkle_roots.sig.json
```

## Verify

```bash
python tools/verify_merkle_signature.py \
  --roots merkle_roots.json \
  --signature merkle_roots.sig.json \
  --public-key public_key.pem
```

---

# Algorithm Choice

Preferred:
- ed25519

Reasons:
- Deterministic signatures
- Small key size
- Modern security
- No padding complexity

---

# Exit Codes

Sign CLI:
- 0 = success
- 4 = signing failure

Verify CLI:
- 0 = valid signature
- 5 = signature invalid
- 6 = artifact mismatch
- 7 = malformed signature file

---

# Non-Goals

Phase 3.1 does NOT include:
- RFC 3161 timestamping
- Transparency log anchoring
- Network calls
- Key management policy
- KMS/HSM integration

These belong to future phases.

---

# Security Boundary

Phase 3 remains deterministic and immutable.
Phase 3.1 produces additional detached artifacts.

Integrity != Attestation.
They are layered.

If `tree_method_version` changes in future phases, existing signatures
remain cryptographically valid but are semantically bound to the method
metadata embedded in the signed artifact.
