# LuxDepthV6 forensic audit and regression repairs

Audit date: 2026-09-23.
Audited baseline: `0528cbbc2d5e3cab9a2572322984823187f6e061`
(current `origin/main` when the audit began). Implementation branch:
`codex/lux-v6-forensic-audit` in `/private/tmp/tp-lux-v6-forensic-audit`.
Fresh PR checkout: `/private/tmp/tp-lux-v6-forensic-pr` on
`codex/lux-v6-forensic-fixes`, cut from the same fetched clean `origin/main`
baseline. The repairs were independently reviewed and validated again there.

This audit reviewed standalone retained-source planning, source admission,
reconstruction, grading, encoders, semantic replay, completion publication, and
managed request/admission/worker/artifact boundaries. Five defects were
reproduced with failing regression tests and repaired. V3 remains the production
default and rollback boundary. These results do not establish photographic
fidelity, physical depth accuracy, or hosted deployment acceptance.

## Reproduced findings and repairs

| Priority | Finding | Reproduction and effect | Minimal repair |
| --- | --- | --- | --- |
| P1 | Late output mutations escaped standalone completion verification | Change one byte in `baseline.npy`, remove it, or add an undeclared file during final retained-source validation. Both execution and verification previously accepted all three mutations. Execution could create success evidence for an already-corrupt generation. | Rehash the complete replayed output inventory after source and processing validation, then recheck the exact namespace before accepting completion. Six adversarial cases now reject; execution leaves no completion record. |
| P2 | Execution deadline was omitted from retained-source validation | Advance the monotonic clock past a one-second budget after the first source snapshot. Execution read 18 artifacts before noticing the deadline and had already created its output directory. | Pass the combined deadline/cancellation checkpoint into source validation and recheck before creating output. The regression stops after the first snapshot with no output directory. |
| P2 | Saturation introduced chroma into exact neutral RGB | At saturation 2, neutral linear RGB 0.18 acquired channel errors of approximately +2, 0, -6 float32 ULPs; neutral 1 and 16 acquired +1, 0, -4 ULPs. Finite float32-limit neutrals could fail with false overflow. | Preserve exact equal-channel candidates after exposure, white balance, and contrast. Six regressions cover signed zero, signed HDR through float32 limits, multiple saturation settings, and combined controls. |
| P2 | Managed request and schema validation accepted structurally impossible output budgets | Budgets at or below 17,825,793 bytes passed preview validation despite the core composite parser requiring the 17 MiB envelope reservation plus at least one byte for each stage. | Match the HTTP field and packaged V5 execution-plan schema to the existing core minimum of 17,825,794 bytes. Invalid budgets fail before admission. Actual stage/image/publication reservations still require more than this structural minimum. |
| P2 | Positive subnormal depth could normalize to NaN | The 1st-to-99th percentile span of a valid positive float32 depth grid can be smaller than the smallest float32 value. The original division rounded that positive denominator to zero and upstream validation aborted. | Use float64 normalization only when the positive span rounds to zero in float32, then retain the float32 output contract. Ordinary-depth arithmetic is unchanged. |

Sources and regression anchors:

- [Artifact verification](../../src/transformation_portal/lux_depth_v6/evidence.py)
  and [late mutation regressions](../../tests/lux_depth_v6/test_publication_integrity.py).
- [Execution lifecycle](../../src/transformation_portal/lux_depth_v6/pipeline.py)
  and `test_deadline_interrupts_source_validation_before_output_creation` in
  [pipeline tests](../../tests/lux_depth_v6/test_pipeline.py).
- [Color grading](../../src/transformation_portal/lux_depth_v6/color.py) and
  [neutral saturation regressions](../../tests/lux_depth_v6/test_color.py).
- [Shared depth normalization](../../src/transformation_portal/core/depth_evidence.py)
  and [core depth evidence regressions](../../tests/core/test_depth_evidence.py).
- [Managed request validation](../../src/transformation_portal/portal/photography_v6_jobs.py),
  [packaged plan schema](../../src/transformation_portal/schemas/execution/plan.v5.schema.json),
  and [managed API regressions](../../tests/orchestrator/test_managed_v6_api.py).

The neutral comparison was rendered and inspected at
`/private/tmp/tp-lux-v6-audit-neutral-chroma.png`. Its plot measures numerical
error; it does not claim that a few float32 ULPs are visibly distinguishable.
The repaired neutral samples have zero error and preserve their exact bytes.

## Validation evidence

Commands run from the isolated worktree, using the existing repository core
virtual environment. Full successor execution requires permission for local
process inspection on macOS.

| Command | Result |
| --- | --- |
| `make test-lux-depth-v6-contract` | 1,096 passed; V6, V5, V4, depth-evidence, execution-plan, Materials and reconstruction audit contracts |
| `make test-lux-depth-v6-managed-contract` | 144 passed; HTTP controls, managed adapters, dispatch, execution and external-worker contracts |
| `make test-lux-depth-v6-managed-services` | 6 passed with real Postgres/Redis, external workers, fenced publication and HTTP download; inference/runtime materialization fixture-controlled |
| `PYTHONPATH=src make ci-quick` | Passed |
| `PYTHONPATH=src make test-fast` | 77 passed |
| `PYTHONPATH=src make validate-ci` | Passed |
| `PYTHONPATH=src make check-documentation-catalog test-documentation-contract check-doc-heading-links` | Passed; 279 documentation tests |
| `PYTHONPATH=src .venv/bin/python -m mypy --config-file=mypy.ini src/transformation_portal/core/depth_evidence.py src/transformation_portal/lux_depth_v6 src/transformation_portal/portal/photography_v6_jobs.py` | Passed; 15 source files |
| `PYTHONPATH=src .venv/bin/python scripts/governance/check_docs_structure.py --all` | Passed |
| `PYTHONPATH=src make pre-commit` | Passed, including formatting, markers, governance and gitleaks |
| `PYTHONPATH=src .venv/bin/python -m pytest -q tests/lux_depth_v4/test_backend.py tests/lux_depth_v4/test_raw.py` | 47 passed outside the sandbox |

The first baseline successor run had 1,069 passes and nine subprocess failures.
All nine involved macOS `psutil`/`sysctl` process-inspection permission errors.
The dedicated subprocess rerun and the complete post-fix gate passed outside
the sandbox. These were environment failures; no subprocess contracts were
weakened or skipped.

## Preserved contracts and remaining limits

- Routes, response envelopes, selectors, artifact names, inference defaults,
  resource defaults, V3 production selection, and source/output separation are
  unchanged. The schema minimum now matches a pre-existing core restriction.
- Processing hashes bind the changed implementation. Previously prepared plans
  must not execute silently under new code; retain the matching implementation
  and dependencies to verify historical generations, or prepare a new run.
- The extra output integrity pass reads each product once more. It is bounded
  by the frozen artifact sizes and deadline checkpoints. No representative
  large-photograph throughput benchmark was performed for this added I/O.
- Standalone verification remains a point-in-time check of protected files, not
  filesystem immutability against arbitrary concurrent writers after return.
  Managed publication additionally verifies bytes while staging and uses its
  existing immutable generation and dispatch-fence controls.
- Deadlines are cooperative between source/product operations; they do not
  preempt a NumPy, SciPy, or encoder call already in progress. The V5 parent
  semantic verifier also completes its current replay before the next V6
  cancellation checkpoint.
- Independent inspection found no additional actionable issue in native-depth
  validity/support, calibration, alpha abstention, deterministic product replay,
  server-owned runtimes, tenant reauthorization, physical binding checks,
  publisher-policy drift, or dispatch fences. Passing tests establish their
  exercised local contracts, not an exhaustive security proof.
- Native model execution, representative interior/exterior photography,
  calibrated color references, physical-depth ground truth, and hosted browser
  deployment were not rerun by this audit. Earlier dated trial evidence is
  separate and does not establish current production acceptance.
