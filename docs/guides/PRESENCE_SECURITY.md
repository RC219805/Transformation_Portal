# Presence Security

Presence Security v1.2 is an active Transformation Portal package for
sessionized Presence Compiler parameters, controlled countermeasures, manifest
watermarking, and anchor payload generation.

## Runtime Surface

- Package: `transformation_portal.presence_security`
- Module CLI: `.venv/bin/python -m transformation_portal.presence_security`
- Console script: `.venv/bin/presence-security`
- Config: `config/presence_security/v1_2/`
- Schemas: `docs/schemas/presence/`
- Examples: `docs/contracts/examples/tp.presence.*.example.json`
- Reference specs: `docs/projects/presence/`
- Contract skeleton: `docs/contracts/presence/PresenceCompiler.sol`

## Quickstart

```bash
.venv/bin/presence-security params \
  --session "demo-session" \
  --locale US_EN

.venv/bin/presence-security anchor \
  --manifest docs/contracts/examples/tp.presence.manifest.v1_2.example.json \
  --hero ./hero.jpg \
  --web ./web.jpg \
  --out ./anchor_payload.json

.venv/bin/presence-security watermark \
  --image ./hero.jpg \
  --manifest docs/contracts/examples/tp.presence.manifest.v1_2.example.json \
  --session "demo-session" \
  --mode dct \
  --out ./hero_watermarked.jpg
```

## Maintainer Notes

- Parameter derivation is deterministic per session key and locale fallback is
  fail-closed to `US_EN`.
- Anchor payload hashes use SHA3-256 for the manifest, each supplied asset, and
  the combined hero+web asset bundle.
- Watermark helpers preserve the original LSB and DCT embedding behavior.
- The Solidity contract is a reference skeleton with an owner-gated license
  setter; production deployment still requires the target auth and treasury
  model.
- Licensing and roadmap notes are project references, not legal guidance.
