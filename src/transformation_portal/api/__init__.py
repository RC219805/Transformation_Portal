"""Typed FastAPI request/response models for the orchestrator HTTP edge.

This package exists so the wire contracts that `app.py` currently expresses as
ad-hoc `Dict[str, Any]` shapes have a typed, mypy-checkable, OpenAPI-renderable
counterpart. See `docs/architecture/` and the Phase 1.2 plan in PR #1558 for
the migration sequence.

PR A (this introduction) ships the foundation — envelope/error models — without
touching any route in `app.py`. Subsequent PRs (B–E) progressively wire
`response_model=` on individual route groups.
"""
