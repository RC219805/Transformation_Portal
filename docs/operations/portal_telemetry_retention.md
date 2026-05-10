# Portal Telemetry Retention Runbook

Use `tools/portal_telemetry_retention.py` to review approved raw portal telemetry JSONL sinks, delete them when the pilot owner and reviewer approve disposal, and emit deterministic deletion evidence. This tool governs retention evidence only; aggregate rollout evidence still comes from `tools/portal_modernization_evidence.py`.

## Dry Run

Run dry-run mode first and attach the generated JSON to the pilot evidence packet.

```bash
python3 tools/portal_telemetry_retention.py \
  --pilot-owner "RC219805" \
  --pilot-end-date 2026-05-09 \
  --reviewer "RC219805" \
  --sink-path /absolute/path/to/portal-rum.jsonl \
  --sink-path /absolute/path/to/portal-events.jsonl \
  --evidence-out /absolute/path/to/deletion-evidence.json \
  --dry-run
```

Dry-run mode records path existence, prior size, prior modification time, retention deadline, and repo/public/static path classification. It never deletes raw logs.

## Delete

Delete mode requires the explicit confirmation phrase `DELETE-PORTAL-TELEMETRY-RAW-LOGS`.

```bash
python3 tools/portal_telemetry_retention.py \
  --pilot-owner "RC219805" \
  --pilot-end-date 2026-05-09 \
  --reviewer "RC219805" \
  --sink-path /absolute/path/to/portal-rum.jsonl \
  --sink-path /absolute/path/to/portal-events.jsonl \
  --evidence-out /absolute/path/to/deletion-evidence.json \
  --delete \
  --confirm-delete DELETE-PORTAL-TELEMETRY-RAW-LOGS
```

Evidence is written after deletion attempts and includes only path metadata, deletion timestamps, owner, reviewer, mode, and summary counters. It does not preserve raw JSONL records or content hashes.

## Placement And Evidence Expectations

- Raw logs must stay outside the repository, outside public / static directories, access-restricted, and excluded from CI artifacts.
- Missing sink paths are represented in evidence without failing the run.
- Relative paths, directories, glob-like paths, and symlinks are rejected.
- The retention deadline is `pilot-end-date + 14 calendar days`.
- Preserve only aggregate evidence after raw-log deletion. Use `tools/portal_modernization_evidence.py` for aggregate portal modernization evidence before deleting raw sinks.
