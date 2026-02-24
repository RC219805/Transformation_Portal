# Archive Manifest Phase 3.2 - Detached RFC 3161 Timestamp Anchoring

## Status
Implemented (Phase 3.2 - Detached Timestamp Layer)

## Context

Phase 3 provides deterministic hash and Merkle integrity artifacts.
Phase 3.1 provides detached cryptographic attestation of integrity artifacts.

Phase 3.2 adds detached timestamp anchoring so attested artifacts can carry
evidence of existence-at-time without mutating prior artifacts.

This layer remains outside deterministic integrity generation and does not
change Phase 3 or Phase 3.1 output bytes.

---

# Design Principles

1. Timestamping is detached and additive.
2. Existing artifacts are immutable inputs.
3. No Phase 3 or 3.1 file is rewritten.
4. Timestamping is not part of default deterministic CI gates.
5. Timestamp responses are stored as detached `.tsr` artifacts.
6. Artifact binding is explicit by filename and message imprint digest.

---

# Timestamp Target Scope

Phase 3.2 supports detached timestamp requests over exact bytes of:

- `merkle_roots.json` (`--roots`)
- `merkle_roots.sig.json` (`--signature`)

This allows operators to anchor either:
- deterministic integrity artifact bytes, or
- detached attestation bytes from Phase 3.1.

---

# RFC 3161 Request Model

The CLI emits an RFC 3161 `TimeStampReq` with:

- `version = 1`
- `messageImprint.hashAlgorithm = sha256`
- `messageImprint.hashedMessage = sha256(target_bytes)`
- `nonce` (user-supplied or random 128-bit)
- optional `certReq = true` when `--cert-req` is set

Requests are posted as:

- `Content-Type: application/timestamp-query`
- `Accept: application/timestamp-reply`

---

# Response Handling Contract

The CLI enforces:

- DER-encoded top-level `TimeStampResp`
- `pkiStatus` extracted from `PKIStatusInfo`
- success only for `pkiStatus` in `{0, 1}`
- granted responses must include a `timeStampToken`
- media type must be `application/timestamp-reply` when present

The raw response bytes are written directly to the detached output `.tsr` path.

---

# CLI Interface

Timestamp `merkle_roots.json`:

```bash
python tools/timestamp_merkle_signature.py \
  --roots /path/to/merkle_roots.json \
  --tsa-url https://tsa.example.com \
  --out /path/to/merkle_roots.tsr
```

Timestamp `merkle_roots.sig.json`:

```bash
python tools/timestamp_merkle_signature.py \
  --signature /path/to/merkle_roots.sig.json \
  --tsa-url https://tsa.example.com \
  --out /path/to/merkle_roots.sig.tsr
```

Optional deterministic nonce and cert request:

```bash
python tools/timestamp_merkle_signature.py \
  --signature /path/to/merkle_roots.sig.json \
  --tsa-url https://tsa.example.com \
  --out /path/to/merkle_roots.sig.tsr \
  --nonce 42 \
  --cert-req
```

---

# Exit Codes

- `0` = timestamp response written successfully
- `8` = timestamp request/setup failure (input, I/O, URL, HTTP transport)
- `9` = invalid or rejected timestamp response

---

# Non-Goals

Phase 3.2 does not include:

- automatic CI timestamping
- TSA trust policy management
- certificate path validation of TSA signer
- transparency log anchoring
- evidence bundle canonicalization (Phase 3.4 scope)

---

# Trust Boundary

- Phase 3 remains deterministic integrity authority.
- Phase 3.1 remains detached cryptographic attestation authority.
- Phase 3.2 adds detached existence-at-time evidence.

Together, these layers provide identity + integrity + attestation + time
without collapsing contract boundaries.
