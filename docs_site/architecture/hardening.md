# Hardening Layer

The hardening layer provides guardrails (input validation, output root enforcement, stamping, manifesting)
without changing core pipeline behavior unless explicitly enabled.

Recommended usage:
- Use hardened wrappers for production runs and benchmarks.
- Keep a policy file for operational constraints (max input size, allowed roots).

