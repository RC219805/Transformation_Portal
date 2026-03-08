# APEX CI Integration Guide

**Version:** 1.0.0
**Date:** 2026-02-07
**Audience:** DevOps Engineers, CI/CD Maintainers

## Overview

This guide explains how to integrate APEX end-to-end workflow into GitHub Actions CI/CD pipelines.

## Prerequisites

- Python 3.11+
- Transformation Portal installed (`pip install -e .`)
- APEX ledger database accessible to CI runners
- Multi-zone infrastructure (optional, defaults to "local")

---

## Basic CI Workflow Structure

### 1. Matrix Run (Parallel Execution)

```yaml
name: APEX Performance Matrix

on:
  pull_request:
  push:
    branches: [main]

jobs:
  apex_matrix:
    name: APEX Run (${{ matrix.workflow_version }} / ${{ matrix.zone }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false  # Continue running even if one zone fails
      matrix:
        workflow_version: [v1, v2]
        zone: [local]  # Expand to [us-west-2a, us-west-2b, us-east-1a] for multi-zone

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install -r requirements-ci.txt

      - name: Run APEX
        env:
          APEX_ZONE: ${{ matrix.zone }}
        run: |
          python scripts/apex_matrix_runner.py \
            --run-id "${{ github.sha }}" \
            --commit-sha "${{ github.sha }}" \
            --workflow-versions "${{ matrix.workflow_version }}" \
            --zones "${{ matrix.zone }}" \
            --output-dir ./apex_results \
            --ledger-db ./apex_performance.db \
            --dry-run  # Remove for actual runs

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: apex-results-${{ matrix.workflow_version }}-${{ matrix.zone }}
          path: apex_results/
          retention-days: 30
```

### 2. Aggregation and Gating

```yaml
  apex_gate:
    name: APEX Gate
    needs: apex_matrix
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Download All Results
        uses: actions/download-artifact@v4
        with:
          path: apex_results/

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install -r requirements-ci.txt

      - name: Aggregate Results
        run: |
          # Combine all observations
          python scripts/apex_aggregate_results.py \
            --results-dir apex_results/ \
            --ledger-db apex_performance.db

      - name: Evaluate Gate
        run: |
          python scripts/apex_evaluate_gate.py \
            --run-id "${{ github.sha }}" \
            --ledger-db apex_performance.db \
            --mode enforce \
            --worst-zone-p95-threshold 15.0 \
            --max-regression-threshold 0.15

      - name: Generate PR Comment
        if: github.event_name == 'pull_request'
        run: |
          python scripts/apex_pr_comment.py \
            --run-id "${{ github.sha }}" \
            --commit-sha "${{ github.sha }}" \
            --ledger-db apex_performance.db \
            --output apex_comment.md

      - name: Post PR Comment
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const fs = require('fs');
            const comment = fs.readFileSync('apex_comment.md', 'utf8');

            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: comment
            });
```

---

## Multi-Zone Configuration

### Kubernetes Deployment

Add Downward API to pod spec:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: apex-runner
spec:
  containers:
  - name: runner
    image: transformation-portal:latest
    env:
    - name: KUBE_NODE_ZONE
      valueFrom:
        fieldRef:
          fieldPath: metadata.labels['topology.kubernetes.io/zone']
    volumeMounts:
    - name: podinfo
      mountPath: /etc/podinfo
  volumes:
  - name: podinfo
    downwardAPI:
      items:
      - path: "zone"
        fieldRef:
          fieldPath: metadata.labels['topology.kubernetes.io/zone']
```

### AWS EC2 Deployment

Zone is automatically detected via IMDSv2. No configuration needed.

### Manual Override

```yaml
env:
  APEX_ZONE: custom-zone-name
```

---

## Shadow Mode (V2 Rollout)

During V2 rollout, use shadow mode to collect data without blocking:

```yaml
- name: Evaluate V2 Gate (Shadow Mode)
  run: |
    python scripts/apex_evaluate_gate.py \
      --run-id "${{ github.sha }}" \
      --ledger-db apex_performance.db \
      --workflow-version v2 \
      --mode shadow  # Warns but doesn't block
```

After 30 days of shadow mode:
1. Review V2 performance vs V1
2. Adjust V2 thresholds if needed
3. Switch to `warn` mode
4. After another 30 days, switch to `enforce`

---

## Ledger Management

### Initialization

```yaml
- name: Initialize Ledger
  run: |
    python -c "from transformation_portal.metrics.ledger import PerformanceLedger; \
               PerformanceLedger('apex_performance.db')"
```

### Pruning Old Data

Add scheduled workflow:

```yaml
name: APEX Ledger Maintenance

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  prune:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Prune Old Entries
        run: |
          python -m transformation_portal.metrics.ledger prune \
            --ledger-db apex_performance.db \
            --days-to-keep 90
```

---

## Artifact Storage

### Local Storage (Small Teams)

Store ledger as CI artifact:

```yaml
- name: Upload Ledger
  uses: actions/upload-artifact@v4
  with:
    name: apex-ledger
    path: apex_performance.db
    retention-days: 90
```

### Persistent Storage (Production)

Mount persistent volume or use cloud storage:

```yaml
- name: Download Ledger from S3
  run: |
    aws s3 cp s3://my-bucket/apex_performance.db ./apex_performance.db

- name: Upload Ledger to S3
  if: always()
  run: |
    aws s3 cp ./apex_performance.db s3://my-bucket/apex_performance.db
```

---

## Security Considerations

### Secrets Management

```yaml
env:
  APEX_ZONE: ${{ secrets.APEX_ZONE }}
  LEDGER_ENCRYPTION_KEY: ${{ secrets.LEDGER_ENCRYPTION_KEY }}
```

### Artifact Access Control

Restrict artifact access to repository collaborators:

```yaml
- name: Upload Results
  uses: actions/upload-artifact@v4
  with:
    name: apex-results
    path: apex_results/
    retention-days: 30
    # Artifacts are only accessible to repository collaborators
```

---

## Troubleshooting

### Gate Blocks Unexpectedly

1. Check gate logs for specific violation
2. Query ledger for historical baseline:
   ```bash
   python -m transformation_portal.metrics.ledger query \
     --ledger-db apex_performance.db \
     --workflow-version v1 \
     --min-days 30 \
     --output baseline.json
   ```
3. Compare current vs baseline manually
4. If threshold is too strict, escalate to Architect for review

### Matrix Job Failures

1. Check `--continue-on-error` flag to prevent early exit
2. Review failed job logs
3. Re-run failed jobs only:
   ```bash
   python scripts/apex_matrix_runner.py \
     --workflow-versions v1 \
     --zones failed-zone \
     --output-dir ./apex_results
   ```

### Ledger Corruption

1. Restore from backup (S3/artifact)
2. Rebuild from raw capsules:
   ```bash
   python scripts/apex_rebuild_ledger.py \
     --capsules-dir ./apex_results \
     --ledger-db apex_performance.db
   ```

---

## Performance Optimization

### Reduce CI Runtime

- Use sparse zones for PRs (1-2 zones)
- Use full matrix for main branch
- Cache dependencies

```yaml
- name: Cache Dependencies
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

### Parallel Execution

Matrix jobs run in parallel by default. Ensure enough GitHub Actions runners.

---

## Monitoring and Alerting

### Failed Gate Notifications

```yaml
- name: Notify on Gate Failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "APEX Gate BLOCKED PR #${{ github.event.pull_request.number }}: ${{ github.event.pull_request.title }}"
      }
```

### Dashboard Integration (Future)

- Export metrics to Prometheus/Grafana
- Visualize zone×bucket heatmap
- Alert on threshold violations

---

## Migration from Legacy System

If migrating from CSV-based tracking:

1. Run both systems in parallel for 30 days
2. Validate results match
3. Disable CSV export
4. Decommission legacy scripts

---

## References

- **End-to-End Architecture:** `docs/architecture/APEX_END_TO_END_ARCHITECTURE.md`
- **ADR-025:** `docs/architecture/decisions/ADR-025-APEX-end-to-end.md`
- **APEX Executive Summary:** `docs/historical/APEX_EXECUTIVE_SUMMARY_20260207.md`

---

**Maintained By:** DevOps Team
**Review Cycle:** Quarterly
**Next Review:** 2026-05-07
