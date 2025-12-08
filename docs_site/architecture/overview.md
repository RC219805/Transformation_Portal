# Architecture Overview

This project is designed around **modular pipelines**, with additive layers for:
- **Hardening** (security + reproducibility + guardrails)
- **Observability** (Prometheus metrics + structured logs + request correlation)
- **Validation** (baseline comparisons + multi-metric scoring)

## Design principles

- **Additive by default**: new capability should not break existing usage.
- **Low-cardinality metrics**: no request IDs or filenames in Prometheus labels.
- **Deterministic stamping**: config hashing + commit stamping for reproducibility.
- **Operational safety**: input validation and output path constraints when deployed.

