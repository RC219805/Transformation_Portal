# ADR-049: Plugin Manifest Trust Boundary

**Status:** Accepted
**Date:** 2026-05-30
**Owners:** Repository Architect, Portal Steward
**Related:** [Portal audit backlog M-1](../governance/audit/PORTAL_AUDIT_2026-05-18_backlog.md#m-1-sandbox-or-sign-plugins-before-broader-use)

## Context

External plugins are already secure by default because discovery outside the
built-in plugin root requires explicit opt-in. Once enabled, plugin packages
run in the Transformation Portal process. That is acceptable for controlled
local extension, but it needs a provenance gate before broader use.

The near-term requirement is a least-invasive trust boundary for in-process
plugins. Marketplace or multi-tenant execution requires a separate process or
service boundary and a smaller API surface; that is intentionally deferred to a
future architecture decision.

## Decision

Use signed `plugin.json` manifests plus a configured trust set for external
plugin packages.

- External plugin loading remains opt-in through
  `TRANSFORMATION_PORTAL_ENABLE_EXTERNAL_PLUGINS` or the explicit loader
  parameter.
- When `TRANSFORMATION_PORTAL_PLUGIN_TRUST_STORE` or
  `PluginLoader(plugin_trust_store_path=...)` is configured, external package
  manifests must verify before any plugin module import occurs.
- The initial verifier uses deterministic HMAC-SHA256 over canonical manifest
  JSON, with `signature_algorithm`, `signature_key_id`, and `signature`
  excluded from the signed payload.
- Built-in plugins are trusted by repository review and are not required to
  carry signatures.
- External single-file plugins are skipped while a trust store is configured,
  because they do not carry a manifest boundary.

## Trust Store Shape

```json
{
  "keys": {
    "local-dev": "shared-secret"
  }
}
```

This is an in-process provenance check, not a sandbox. The trust store should
be readable only by operators who are allowed to authorize local plugin code.

## Consequences

- Existing local plugin tests and unsigned external plugins keep their current
  behavior when no trust store is configured.
- Operators can require signed manifests without changing the public plugin
  interface.
- Unsigned or tampered external packages fail before import, which bounds the
  highest-risk side effect.
- Worker-process isolation remains deferred for marketplace or multi-tenant
  plugin distribution.

## Validation

- Signed external `plugin.json` packages load with a matching trust store.
- Unsigned external package manifests are rejected when trust is configured.
- Tampered manifests are rejected before plugin module import.
