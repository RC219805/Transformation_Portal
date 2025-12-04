# Archived Workflows

This directory contains deprecated workflow files that have been superseded by the consolidated CI/CD pipeline.

## Archived Files

| File | Replaced By | Reason |
|------|-------------|--------|
| `build.yml.archived` | `ci-consolidated.yml` | Consolidated for 40-60% CI time reduction |
| `python-app.yml.archived` | `ci-consolidated.yml` | Merged into unified pipeline with intelligent test selection |

## Why Keep Them?

These files are preserved for:
- Historical reference
- Rollback capability if needed
- Understanding the evolution of the CI/CD pipeline

## Current Active Workflow

The primary CI/CD workflow is now **`ci-consolidated.yml`** which provides:
- Intelligent change detection (skips irrelevant jobs)
- Unified lint, test, and build pipeline
- Matrix testing across Python 3.10, 3.11, 3.12
- RAG system validation integration
- 40-60% reduction in CI execution time

## Migration Date

Archived: December 2025
