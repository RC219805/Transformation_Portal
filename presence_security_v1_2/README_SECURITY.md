# Presence Security v1.2 — Implementation Kit

This package contains code and artifacts to harden the Presence Compiler and platform.

## Contents
- presence_params.py — sessionized parameter derivation (obfuscation)
- watermarking.py — LSB/DCT watermark embed/extract
- countermeasures.py — controlled randomness & anti-fingerprinting
- PresenceCompiler.sol — license + manifest anchoring skeleton
- presence_cli_v1_2.py — CLI extensions (anchor, watermark, params)
- LICENSE_TIERS.md — tiered licensing outline (non-legal; consult counsel)
- CERTIFICATION.yml — Bronze/Silver/Gold verification levels
- TRUST_REGISTRY_SCHEMA.json / EXAMPLE.json — registry blueprint
- RANDOMIZATION_CONFIG.yml — ranges for randomized parameters
- SECURITY_ROADMAP.md — Week/Month/Quarter plan

## Quickstart
```bash
# params
python presence_security_v1_2/presence_cli_v1_2.py params --session "demo-session" --locale US_EN
# anchor
python presence_security_v1_2/presence_cli_v1_2.py anchor --manifest presence_manifest_example.json \
  --hero In-Command_In-Conversation_2400x3000.jpg \
  --web In-Command_In-Conversation_1065x1330.jpg
# watermark (DCT)
python presence_security_v1_2/presence_cli_v1_2.py watermark --image In-Command_In-Conversation_2400x3000.jpg \
  --manifest presence_manifest_example.json --session "demo-session" --mode dct --out hero_wm.jpg
```

## Notes
- The Solidity contract is a skeleton; integrate with your auth/treasury model and add ownership/role controls.
- Watermarking thresholds and DCT positions are conservative; tune after lab testing for your JPEG pipeline.
- This kit is a technical reference, not legal advice.
