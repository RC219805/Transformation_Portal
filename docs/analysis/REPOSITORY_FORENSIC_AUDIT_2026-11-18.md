# Repository forensic code audit and repairs

Audit date: 2026-11-18.
Audited baseline: `9e2e25e7869c57cc2f8d189851dfe3db8f183d0c` (`origin/main` when the audit began).
Implementation branch: `claude/eloquent-galileo-smk78g`, 430 commits ahead of the baseline at the time of writing (`a915b7202de0`).

This is dated audit evidence, not live guidance. It records what a whole-repository review found, what was
repaired on the branch above, and what was left for an owner decision. It does not establish photographic
fidelity, physical depth accuracy, hosted deployment acceptance or behaviour against live managed services.

## Method

- Sixteen subsystem reviews covered the Python package, the FastAPI origin, the governed tools and scripts,
  the managed front door and the Cloudflare worker. A finding is listed as confirmed only when a script
  reproduced it; findings that could not be executed here (model weights, Ray, a browser, deployed
  configuration) are marked plausible.
- Each repair is one commit that names its finding, changes the smallest seam that removes the root cause and
  adds a regression test that fails without the change.
- Repairs were then reviewed independently and adversarially; review follow-ups and reverts are separate commits.
- A finding was deferred, not repaired, when the repair would change governed deliverable bytes, plan, evidence
  or cache identities of runs that succeed today, a Protocol, schema or CLI contract, behaviour pinned by a
  test, document or ADR, or would need a threshold that had to be invented.

## Result

374 findings: 300 fixed, 43 partly fixed, 20 deferred for an owner decision, 0 rejected on re-verification, 10 plausible but not confirmed, 0 open, 1 tracked under another row.

| Subsystem | Findings | Critical/high | Fixed | Partly | Deferred | Open | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FastAPI origin and portal helpers (app.py, portal/) | 15 | 2 | 15 | 0 | 0 | 0 | 0 |
| Durable state: orchestrator/, storage/, events/, hardening/, api/ | 14 | 0 | 14 | 0 | 0 | 0 | 0 |
| Lux Depth V3 orchestrator core | 17 | 2 | 13 | 2 | 1 | 0 | 1 |
| Lux Depth V3 evidence, lifecycle, cache, manifest, I/O, CLI | 14 | 1 | 12 | 1 | 1 | 0 | 0 |
| Lux Depth V3 pixel, inference, segmentation, materials, PBR | 21 | 1 | 17 | 1 | 3 | 0 | 0 |
| Lux Depth V4/V5/V6 and materials_v4 | 11 | 0 | 10 | 1 | 0 | 0 | 0 |
| core/, determinism/, execution_graph/, stage_graph/ | 32 | 3 | 28 | 0 | 2 | 0 | 2 |
| Trust surfaces (tp/, attestation/, plugins/, compliance/, presence_security/) and runtime/ | 26 | 6 | 19 | 2 | 2 | 0 | 3 |
| Governed CLIs and repository scripts (tools/, scripts/) | 20 | 3 | 17 | 2 | 0 | 0 | 1 |
| Managed front door (web/secure-landing) and Cloudflare worker | 11 | 1 | 10 | 0 | 1 | 0 | 0 |
| Depth backends and depth tooling | 26 | 3 | 12 | 11 | 3 | 0 | 0 |
| Ingest, photographic delivery, VLM captioning, recipe pipeline | 23 | 1 | 18 | 3 | 2 | 0 | 0 |
| spatial_ai package | 36 | 5 | 30 | 6 | 0 | 0 | 0 |
| Image and video pipelines, processors, streaming, utils | 48 | 9 | 33 | 7 | 4 | 0 | 4 |
| dashboard/ and rl/ packages | 29 | 6 | 27 | 1 | 1 | 0 | 0 |
| Evaluation, metrics, perceptual, foundation, dev tooling, auxiliary packages | 31 | 4 | 25 | 6 | 0 | 0 | 0 |

## Validation evidence

- Baseline core suite as found: 20,227 passed, 10 failed, 6 errors. All sixteen were test or environment coupling
  (tests that assume a non-root user, tests that need torch without a guard), repaired first.
- Core suite on the branch head above: 22,482 passed, 637 skipped, 0 failed.
- ML tier (Python 3.11, CPU torch): 1,151 passed, 72 skipped, 0 failed (894 passed at the baseline).
- Managed front door (Node 22): npm test 989 passed, 0 failed; npm run build compiled.
- Governance gates: check-json-serialization, check-yaml-governance, check-python-headers, check-piptools-cache, check-requirements-lock-contract, check-dependency-pinning, check-ci-sync, check-portal-asset-budgets, validate-ci, the test-marker audit, the tautological-test ban and the TODO scan pass; black, isort, flake8 and the CI mypy allowlist are clean on every changed file apart from one mypy report in api/docs.py:101 that predates the audit and appears only with a newer local mypy.
- Not exercised here: live Postgres, Redis and S3 backends (their contract suites skip without service URLs),
  GPU and MPS paths, inference with real model weights, browser behaviour of the portal and dashboard pages,
  and the deployed Cloudflare and Vercel configuration.

## Decisions needed

These findings are real but were not repaired, or only in part, because the repair is a contract or product decision.

| Finding | Severity | Where | Decision needed |
| --- | --- | --- | --- |
| depth D-22: PLAUSIBLE — DA2 depth orientation is inverted relative to how it is documented and consumed | high | `depth/models/depth_anything_v2.py:579-580; depth/depth_processor.py:98-99; depth_intellige` | plausible only, needs model weights to confirm DA2 orientation |
| frontdoor F-01: Worker and frontdoor disagree about the public host | high | `cloudflare/transformationportal-worker/src/index.ts:44-46, 77-79, 99-113; web/secure-landi` | plausible integration finding that depends on deployed Cloudflare/Vercel configuration |
| luxc LC-01: dead acceptance gate in CLIP material labelling | high | `lux_depth_v3/segmentation/efficient_sam.py:751-763; lux_depth_v3/segmentation/_cache.py:11` | enabling the CLIP gate needs a calibrated threshold and changes labelling for every run |
| misc M-01: NaN and off-scale scores are clamped to a perfect 1.0 | high | `evals/apex_harness.py:206; evals/metrics.py:400-401; evals/vision_language/llava_schema.py` | f6e0fdb2f. Remaining: Deferred: finite off-scale scores (3, 35) are still clamped to 1.0 because test_llava_schema.py::TestParseVqaResult::test_score_clamping, tests/validation/test_llava_quality_validation.py::test_parse_score_clamping and test_apex_harness.py::test_run_metrics_score_clamping pin clamping 1.5 to 1.0; the owner must decide whether scores above 1 become a parse error (score 0) or keep clamping, and whether compute_quality_gate_pass should also require passes_basic_quality. |
| misc M-02: _to_numpy decides the data range from pixel content | high | `evals/metrics.py:71-75` | 495647ad1. Remaining: Deferred (needs a new threshold or an explicit value-range argument, per the pinned test_high_value_normalization convention): two float images that BOTH overshoot 1.0 are still read as 0-255 data (+48 dB PSNR inflation), and signed/wider integer containers (int64 from Python ints) still follow the content convention because np.iinfo(int64) would turn 0-255 data into ~1e-17 and inflate PSNR far more; the owner must decide between clipping floats, a range threshold or a required value_range parameter. Note for M-03: uint16 masks now go through the uint16 dtype scaling in _to_numpy until the mask helpers binarise |
| pipelines H-01: HDR tone curves applied to display-referred 8-bit input | high | `tp/pipelines/rendering_4k/stages.py:35-113; rendering_4k/pipeline.py:482-483, 518-519` | tone curves on display-referred input is a decision about the look |
| pipelines H-05: 16-bit greyscale inputs load as almost pure white | high | `tp/utils/image_utils.py:25; pipelines/unified_luxury_pipeline.py:789-791; pipelines/pro_pi` | 22adf7ea0. Remaining: Deferred: 32-bit integer TIFFs (mode I) and float TIFFs (mode F) still clip via Pillow because their full scale is not defined by the file; the owner must decide the convention (16-bit vs 32-bit range for int32, [0,1] vs [0,255] for floats; the unified tifffile path already assumes [0,1] floats and min-max stretches other ints). |
| pipelines H-07: the 10-bit "master" chain is squeezed through 8-bit | high | `tp/processors/luxury_video_master_grader.py:543-612, 664-671` | replacing eq/gradfun redefines the grade maths (the look) |
| spatial S-2: seam validator aborts the run for pairs the merger itself decided to merge | high | `spatial_ai/segmentation/tiling/merger.py:332-337; spatial_ai/segmentation/tiling/validator` | d0acbc0f2. Remaining: Deferred, owner decision: a pair that still disagrees after the retry (IoU >= 0.35, continuity < 0.75) aborts the image as before; the options are not merging such pairs in the merger, aligning seam_discontinuity_threshold with instance_merge.iou_threshold, or downgrading a validation failure to a recorded warning, and each needs a new threshold or changes masks of runs that succeed today with validation disabled. Follow-up outside this cluster: lux_depth_v3/segmentation/sam2.py records the configured overlap as sam2_runtime.tiling.effective, so a run that only succeeds on a retry reports e.g. 256 in its run ca |
| trust T-06: ImmutableLedger: a failed append bricks verification, concurrent handles fork the chain, and tampering verifies | high (latent) | `runtime/ledger.py:214` | 6764797e9. Remaining: Deferred for the owner (ADR-051 already marks ImmutableLedger frozen): the chain is still unkeyed and the signer still does not sign entry hashes, so a writer holding the issuer key, or one who recomputes all hashes, can rewrite history. Deciding what to sign, key pinning for verify(), and the policy for unsigned legacy entries is a keyed-format decision. |
| luxa F3: the documented APEX v2-auto recovery is unreachable | medium-high | `orchestrator.py:4694-4703; config.py:738-747` | docs-versus-code contract question (APEX v2-auto recovery) |
| core CORE-05: the signature covers only manifest_hash + root_hash. issuer, signed_at, metadata and certificate_id are unauthenticated. verify() — the call the modul | medium | `core/security/signing.py:298-302` | Reproduced at 1208cab: core/security/signing.py signs only (manifest_hash + root_hash), so issuer/signed_at/metadata/certificate_id can be rewritten and still verify, and CertificateVerifier.verify() accepts any certificate self-signed by the key embedded in it; no code changed. Authenticating the other fields changes the signed bytes of certificates that verify today (cluster guidance: defer), and making the pinned-key check the default breaks the documented verify(manifest_json, cert) API pinned by tests/security/test_signing_tls_helpers.py::test_certificate_signer_round_trip and the module docstring. Owner must decide: (a) a vers |
| delivery D-02: GPS hemisphere lost; shutter speed and dimensions never populated | medium | `ingest/provenance.py:362-363` | a repair was written and then reverted in review (8ea305de1): REVERTED by b76db7f (explicit undo; a plain git revert conflicts with the later D-13 tests in tests/ingest/test_provenance_capture.py). The code itself was correct and its tests fail without the source change, but it is a governed identity change that the cluster definition says to defer: exif.gps_latitude / gps_longitude are file-derived and survive the ingest_v1 normalisation profile, so negating them for South/West captures changes the provenance sidecar bytes, the normalised sidecar sha256 and the batch root hash of every geotagged southern/western-hemisphere file tha |
| delivery D-04: ingest_v1 normalisation keeps volatile exiftool tags | medium | `ingest/normalize_machine_json.py:16-24, :56-60; raw_sidecar.py:24-27` | needs a new ingest normalisation profile id (hashes change) |
| depth D-15: ndarray inputs to DepthBackend.compute() are mangled, differently per backend | medium | `depth/backends/da3.py:1111-1125; depth/backends/synthetic.py:101; depth/backends/depthcraf` | ee387268f. Remaining: Deferred: a float frame with max > 1.0 is still read as [0,255] pixels in every backend (documented DA2/synthetic/DepthCrafter convention, Lux V3 inference instead clips to [0,1]); the owner must decide whether depth backends treat such floats as HDR [0,1] data. |
| depth D-23: DA3 and Depth Pro parents accept an exit-0 worker's depth without validating it; malformed outputs escape the failure classification | medium | `depth/backends/da3.py:1338-1348,1367-1370; depth/backends/depth_pro.py:537-540` | 56fc75e66. Remaining: Deferred: depth shape equality with the echoed input_size, because the legacy DA3 engine returns depth at the model's processing grid (default 504 long edge), so the owner must decide whether the worker echoes the depth grid or the parent resizes/accepts a different grid. |
| depth D-17: Depth Pro subprocess calls have no timeout and inherit the parent's Python environment | medium | `depth/backends/depth_pro.py:339-348; da3.py:483-496; depth/backends/depth_pro_worker.py:38` | ec09d9c8f. Remaining: Deferred: installed-package layouts (no checkout src) still forward the caller's PYTHONPATH because test_license_enforcement.py::test_subprocess_without_repo_checkout_uses_caller_cwd pins it (DA3 drops it); the owner must decide whether installed layouts require transformation_portal installed in the Depth Pro venv, and whether the legacy worker argv mode needs its own licence-acknowledgement flag (low, related). |
| depth D-18: DepthEnsembleBackend.ensure_available() reports ready when no constituent can run | medium | `depth/backends/ensemble.py:852-873; lux_depth_v3/pipeline_coordinator.py:936` | f1345e772. Remaining: Deferred: (1) requiring >=2 *ready* constituents would turn today's successful single-runnable-model ensemble runs into coordinator fallbacks, so the owner must decide whether a degraded one-model ensemble is acceptable (currently warned, not failed); (2) two-model fusion degenerates to a plain weighted mean because both per-pixel confidences are identical, a fusion-algorithm decision that changes ensemble depth bytes. |
| depth D-10: DepthAwareDenoise quantises float images to 8 bits and destroys any frame with a sample above 1.0 | medium | `depth/processors/depth_aware_denoise.py:155-158,187-188; config/default_config.yaml` | c20937553. Remaining: Deferred: the 8-bit quantisation (128 -> 51 levels) and -0.5 LSB truncation bias of ordinary in-range float frames; the owner must decide whether to run cv2.bilateralFilter on float32 (and round) at the cost of changing every current denoise output. |
| depth D-25: default agx / reinhard zone tone curves map SDR white to 0.5 | medium | `depth/processors/zone_tone_mapping.py:231-269; config/default_config.yaml` | default zone tone curves are a product decision about the look |
| depth D-20: DepthProcessor ignores device, silently changes the variant, loads the model twice, reports zero cache stats | medium | `depth/depth_processor.py:72-77; depth/pipeline.py:103-126` | b2ce5ee3e. Remaining: Deferred: the "vitl" comment implies the Large model was intended, but it has always loaded SMALL; the owner must decide whether DepthProcessor should switch to LARGE (changes model, memory and depth output of every current run). |
| depth D-14: depth tools caches retain every full-resolution depth map and mask with a 0 % hit rate | medium | `depth/tools.py:181-182` | cbca04d1e. Remaining: Deferred: load_depth_normalized quantises 16-bit depth to 256 levels via a uint8 resize whenever target_size is given, even at native size (what the batch always passes); the owner must decide whether to skip the round trip at native size and/or resize in mode "F", which changes the output bytes of every ordinary batch run. |
| depth D-24: load_image / save_image choose the value range from the data instead of the dtype | medium | `depth/utils/image_utils.py:74-77` | 50b2b6ef9. Remaining: Deferred: truncation vs rounding in save_image (0.999 -> 254); the owner must decide whether to switch to rint, which changes the bytes of every float image the depth pipeline saves today. Note: float arrays on a 0..255 scale are no longer accepted by save_image (no in-repo caller passes them). |
| luxa F15: on the production path every output byte is read and SHA-256-hashed about 10× per fresh batch and about 14× per resume batch | medium | `probe_prepared_hash_amplification.py; probe_prepared_hash_breakdown.py` | 46e3d195d. Remaining: Deferred to the execution_evidence owner: the preliminary evidence build in _activate_carrier_outcome_projection (2.0x), build+verify in the evidence writer (4.0x) and reuse priming (2.0x) live in execution_evidence.py / the evidence authority; skipping the first prepared run-card emission was not done because the first carrier outcome projection is derived from that emission's outcome, so dropping it can change carrier/evidence bytes and needs an owner decision. |
| luxa F9: v2-auto after total depth failure writes a manifest claiming backend success | medium | `orchestrator.py:4717-4732` | 224715af9. Remaining: Deferred: result["backend"]/fallback_used and the run-card backend_summary still carry the start-up selection, because validators/run_card_validator.validate_run_card_backend_semantics requires a non-empty final_backends_used whenever success_count > 0, and a depth-less v2-auto image has status "ok"; the owner must decide whether such images count as successes with no backend (relax validator rule 1 and report backend=None) or not, together with F3's v2-auto contract decision. |
| luxb F-05: capture_provenance re-probes invariant tools and re-hashes the input for every image | medium | `provenance.py:559-667; provenance.py:130; provenance.py:627` | 1a8f98f44. Remaining: Deferred to the orchestrator.py owner: per-image git rev-parse (a process-wide memo would let the sidecar diverge from the per-orchestrator manifest v3_git_revision pinned by test_orchestrator_run_card_provenance; pass self.v3_git into capture_provenance instead) and the redundant input hash overwritten in prepared runs (add an optional precomputed hash/stat parameter and pass the snapshot's values). |
| luxc LC-07: float truncation in aspect-preserving resize costs a 14-px band | medium | `t_square.py` | changes depth output geometry for ordinary inputs |
| luxc LC-09: PBR numpy path and batched torch path disagree | medium | `lux_depth_v3/pbr.py:62-78, :156-168, :379-400, :437-446, :477-486; t_pbr.py` | 4df743481. Remaining: Deferred: the numpy reference itself computes N=(+dx,+dy,1) against the documented (-dx,-dy,1) and truncates instead of rounding (flat normal 127 not 128); fixing either changes PBR deliverable bytes for every run, so the owner must decide whether to correct the sign/quantisation convention. |
| misc M-05: unavailable LPIPS earns full LPIPS credit | medium | `evals/metrics.py:263-268; evals/benchmark_suite.py:212-222, :295-298` | 91cb49745. Remaining: Deferred: lpips_score itself still returns 0.0 when lpips/torch are missing because tests/evals/test_metrics.py::TestLPIPS::test_lpips_unavailable pins a float return ("returns 0 when unavailable"); the owner must decide whether lpips_score should raise or return None/NaN for an unmeasurable distance. |
| misc M-16: CoreML backend is always "available" | medium | `foundation/hardware_abstraction.py:94-109, :132-176, :217-233, :244-293, :303-305` | a96f706b8. Remaining: Deferred the PLAUSIBLE item: nn.Module arguments are still moved in place by .to(device), so after a fallback the caller's model stays on the fallback device; the owner must decide whether fallback should deep-copy modules (memory cost) or move them back afterwards. |
| misc M-18: MemoryManager: A double deallocate puts the same tensor in the pool twice, so two later allocations are the same object. | medium | `foundation/memory_manager.py:262-288, :117-146, :466-495, :412-430; foundation/substrate.p` | b858a1ec1. Remaining: Deferred: on CPU with profiling off the usage estimate is still 0 (nothing is tracked), so the watermarks cannot fire there; the owner must decide whether byte accounting should run regardless of enable_profiling or whether CPU should read process RSS (psutil, optional dependency). |
| misc M-29: SkyBlender and the ComfyUI SkyGANNode | medium | `comfyui/custom_nodes.py:76-81, :246, :287; atmosphere/sky_blending.py:150, :266, :325-335,` | 5c6c48545. Remaining: Deferred: (1) the mu-law(5000)+gamma tone map lifts every non-sky pixel (sRGB 25->176); the owner must choose an invertible tone curve for LDR sources (e.g. inverse of the 2.2 linearisation with highlight compression only above 1.0), which changes rendered bytes. (2) torch.hub.load("intel-isl/MiDaS", trust_repo=True) runs unpinned remote code behind a catch-all except and treats MiDaS inverse depth as far=1; options: pin a reviewed MiDaS revision/checkpoint via the model lock and load locally, require explicit opt-in, or drop MiDaS for the luma heuristic/a governed depth backend, and invert the depth so sky=far |
| pipelines M-05: DOF mode is about 10× slower than it needs to be | medium | `tp/pipelines/depth_tools.py:320-349, 658-675` | DOF filter choice changes the look |
| pipelines M-06: the LPIPS/perceptual path can never run | medium | `tp/pipelines/quality_feedback_bridge.py:101-119, 343-388; tp/pipelines/rendering_4k/qualit` | 538e3e136. Remaining: Deferred: emitting null instead of 0.0, since QualityMetrics (rendering_4k/types.py:205-208) types them as float, test_rendering_4k_quality.py:72 pins 0.0 for the unassessed case and the quality report JSON serialises them; the owner must also decide whether to restore/repoint a real perceptual assessor module (imports in quality_feedback_bridge.py and rendering_4k/quality.py). |
| pipelines M-19: FFmpeg filter "validation" does not contain file access | medium | `tp/utils/security.py:231, 325-338, 353-381` | 7fd844ae8. Remaining: Deferred: the owner must decide whether to replace this denylist of documented path options with a filter-name allowlist, and whether other additional_args such as extra positional output paths need an option allowlist. |
| rl R11: GraphTransformerMuZero.recurrent_inference ignores the action and the hidden state | medium | `rl/gt_muzero.py:312-336` | conditioning the dynamics on the action is an architecture change |
| spatial R-6: validator counts splats behind the camera and fabricates a 0.0 reprojection error | medium | `spatial_ai/reconstruction/geometric_validator.py:225-226, :182-187, :146` | 2c7b6dd96. Remaining: Deferred: compute_reprojection_error still returns a fabricated 0.0 when points_2d is None, pinned by test_cpu_fill_gaps.py::test_geometric_validator_reprojection_and_coverage_edge_cases (== 0.0) and test_coverage_boost.py::test_compute_reprojection_error; the owner must choose between raising ValueError (points_2d effectively required), returning None (Optional[float] return type), or implementing a real reference-free metric. |
| spatial M-2: a Python call per pixel, repeated per segment on the full image | medium | `spatial_ai/materials/heuristic_fallback.py:177, :272; spatial_ai/orchestration/pipeline.py` | 1c71ac62d. Remaining: Deferred: the pipeline still regenerates full-image heuristic intermediates once per segment; routing it through PBRGenerator._generate_batch_heuristic_shared would replace the per-segment retry / CPU-fallback / skip handling and per-segment metadata timing in orchestration/pipeline.py, so the owner must decide whether the materials stage may batch segments (and how a failure of the shared pass maps onto per-segment error strategies). |
| tools T-11: governance-export verification ignores part of the export | medium | `tools/regulatory_export.py:1397-1422; //attacker.example/verify.sh` | 69a865936. Remaining: Deferred: without that flag a rewritten governance_profile_id still verifies (the export is the only source of the id and the documented verify command does not pass it); the owner must decide whether verification should default to DEFAULT_GOVERNANCE_PROFILE_ID, which would fail today's valid custom-profile exports. |
| trust T-07: a valid ADR-049 manifest signature authorises arbitrary code | medium | `plugins/signing.py:28-37, :70-93; plugin.json; plugins/loader.py:375-388, :434-442` | binding code digests into the ADR-049 manifest is a contract change |
| trust T-08: ReproducibilityCertificate authenticates only manifest_hash:root_hash | medium (latent) | `runtime/certificate.py:192-194, :264-265, :327-328; runtime/execution_manifest.py:128-147;` | what the reproducibility certificate authenticates is a signing redesign |
| core CORE-26: off-by-one: int(width * (long_edge / max_side)) truncates. Used by photographic_delivery/exports.py:50. | low | `core/photographic_export.py:35-36; photographic_delivery/exports.py:50; photographic_deliv` | resize_long_edge feeds hash-pinned photographic export deliverables |
| delivery D-16: VLM proxy is blank for 16-bit grayscale and float sources | low | `vlm_captioning/image_proxy.py:114` | 775bbcca8. Remaining: Deferred: modes I (32-bit int) and F (float) still go through convert("RGB"), which treats them on a 0..255 scale; the owner must decide their full-scale convention (e.g. F in [0,1], or per-image min/max stretch) since today's behaviour is correct for 0..255-scaled data. |
| delivery D-08: recipe validation crashes, or passes recipes that cannot run | low | `config_loader.py:242-245, :324; pipeline_unified.py:231-232, :326-327, :340-341, :1074-107` | 66e943dfd. Remaining: Deferred: output format "exr" still validates but cannot be written by PIL; the schema in utils/recipe_validator.py (pinned by tests/utils/test_recipe_validator.py) lists exr in its enum, so the owner must choose between implementing an EXR writer (optional dependency) or dropping exr from both the schema and config_loader. |
| delivery D-17: vlm/ text heuristics give wrong verdicts | low | `vlm/quality_validator.py:349, :426, :438-464, :520-533; vlm/scene_analyzer.py:191, :311-31` | 8480932e1. Remaining: Left as is: in non-strict detailed mode an overall mean >= 7 still passes with one aspect at 3/10; this is pinned by test_validate_strict_mode_fails_when_any_aspect_fails, so the owner must decide whether a failed aspect should block the relaxed verdict. |
| depth D-09: DepthCache: Evidence: CONFIRMED — repro_depth_cache.py | low | `depth/utils/cache.py` | 140442440. Remaining: Deferred: (1) the legacy namespace=None key still ignores shape/dtype because tests/test_da2_inference.py pins it as md5(image.tobytes()) - the owner must decide whether to drop legacy-key compatibility; (2) ordinary entries are still stored as FP16, so a disk-cache re-run differs from the first run by up to ~2.4e-4 - the owner must decide between lossless (2x disk) and the documented FP16 space saving. |
| depth D-06: smooth_depth(method="bilateral") renormalises to [0,1] and quantises to 8 bits | low | `depth/utils/depth_utils.py:251-261` | 86dc7b1da. Remaining: Deferred: the 8-bit quantisation (255 levels over the span); the owner must decide whether to run cv2.bilateralFilter on float32 data (sigmaColor = edge_preserve * span), which changes every bilateral output today (same open decision as D-10). |
| depth D-07: create_depth_of_field_map blows up at depth 0 | low | `depth/utils/depth_utils.py:497-500` | Reproduced: create_depth_of_field_map returns 1.56e9 px at depth 0 because it treats normalised [0,1] depth/focus as metres (denominator depth*(D-f/1000)+1e-8) and mixes mm and m in the numerator, so its magnitudes are meaningless everywhere, not only at 0. It has no callers, no tests and is not exported from depth.utils; the owner must decide between removing it (fails loudly for any external importer) and defining a metric near/far mapping plus pixel scale (with d clamped to > f), which needs calibration I would have to invent. |
| depth D-26: safe_depth_estimation (depth/models/depth_anything_v2.py:819-836) shrinks the caller's PIL image in place, derives original_size from the shrunk image | low | `segmentation/material_segmenter.py:136-147; depth_intelligence/depth_estimator.py:136-139,` | 425917ebc, 20c91b379, 852e02c9f, f44831c02, 73d57a6db. Remaining: Deferred: the backend get_cache_key implementations (da3/depth_pro/da2/ensemble) still hash tobytes() without shape/mode and DA3's omits geometry recipe and revision; nothing in src calls them, and the owner must decide the DA3 cache identity (recipe/revision fields, which overlaps the governed DA3 runtime identity) before any key format changes. |
| luxb F-14: reconstruction diagnostics report percentiles that are not measured | low | `run_card_integrity.py:554-556` | Per-camera reprojection_rmse/max/p50/p95/p99 in tp.reconstruction_diagnostics.v1 are the global scene RMSE by construction (np.full(points_observed, scene.rmse)); Scene3D carries no per-camera 2D observations, so real residuals cannot be computed in this seam, and emitting null or a "not measured" marker changes the bytes of a run-card-indexed deliverable for every reconstruction that succeeds today (tests/test_verify_run_card_integrity.py fixtures pin numeric values). Owner must decide: either plumb per-point observations from the reconstruction builder, or version the diagnostics contract to mark per-camera fields as unmeasured (n |
| luxc LC-15: depth write statistics include zero sentinels | low | `lux_depth_v3/depth_writer.py:188-209, :61-66` | Reproduced: atomic_write_depth_u16_png_with_stats computes min/mean/std after unsupported and non-finite samples become 0 (stats.min=0.0 vs normalization.source_min=0.36). Deferred because the value reaches a governed artifact for ordinary inputs: the orchestrator passes geometry_result.transform.support() as support_mask, and the 14-multiple model grid leaves unsupported borders on nearly every photo (a 3000x2000 input at target 518 has 156000 unsupported pixels), so the stats land as "scaling" in the indexed <stem>_depth_metadata.json sidecar of runs that succeed today. The owner must decide whether depth stats should describe the |
| luxnext LN-11: DA3 and RAW workers inherit the parent's full environment | low | `lux_depth_v4/backend.py:45-65; raw.py:96` | 976506eb4. Remaining: Deferred: a V6-style allow-list that also drops loader-injection (LD_PRELOAD, DYLD_*) and behaviour switches (PYTORCH_ENABLE_MPS_FALLBACK, HF_ENDPOINT); the owner must decide which variables DA3/torch/MPS and the RAW decoder legitimately need and whether to record them in runtime-identity evidence. |
| pipelines L-01: adjust_contrast_saturation pivots "contrast" on the per-pixel channel mean, so it is a second saturation multiplier and does nothing to neutral tones | low | `tp/pipelines/lux_render_pipeline.py:692-705` | contrast pivot is a decision about the look |
| pipelines L-03: quantisation by truncation, not rounding; darkening accumulates across stages | low | `pro_pipeline.py:646, 676, 709, 746, 775, 821, 825; unified_luxury_pipeline.py:1315, 1388, ` | 5d30146e8. Remaining: The pro, unified luxury and 4K pipelines truncated float->uint8/uint16 conversions, darkening output ~0.5 level per stage (5-stage pro run on grey noise: mean 127.57 -> 125.56, now 127.58); a new utils.image_utils.quantize_unit_float (clip + round) replaces all those conversions, with regression tests in test_image_utils, test_pro_pipeline, test_unified_luxury_pipeline and test_rendering_4k_pipeline_smoke. Not done: processors/material_response/engine.py:268 (other cluster's file) still truncates, and stages still exchange 8-bit PIL images rather than float ("keep float between stages" is an interface change th |
| pipelines L-04: --format short flag is a bare - (meant -f); --pattern and --workers are ignored | low | `pro_pipeline.py:950, 1028, 1021, 1033` | 5e300db4f. Remaining: Deferred: --workers is still unused because batch_process is sequential; its help text and docs now say so, and the owner must decide whether to implement parallel batch processing (needs thread-safe stats and a memory budget) or retire the option. |
| pipelines L-12: a missing custom schema path silently falls back to the built-in schema; the no-jsonschema fallback has a different stage list | low | `utils/recipe_validator.py:283-287, 340-351` | 04392254e. Remaining: Deferred: a missing custom schema_path still silently uses the built-in schema because tests/utils/test_recipe_validator.py::test_nonexistent_schema_path_uses_default deliberately pins that fallback; the owner must decide whether a missing schema_path should raise (fail closed) and update that test accordingly. |
| pipelines L-16: a script inside the package: bare sibling import, plus mkdir and image processing at import time | low | `enhancers/enhance_pool_aerial.py:6, 18-46` | d90a225a4. Remaining: Deferred: main() still cannot run because _assign_full_image, _cluster_stats, _downsample_image and assign_materials no longer exist in board_material_aerial_enhancer (and enhance_aerial's signature differs); the owner must decide whether to port the script to update_enhance_aerial's API or retire it. |
| rl R16: smaller defects, each confirmed unless marked | low | `rl/muzero_trainer.py:180; rl/ma_comm.py:137-144, 158-162; rl/ma_optimize.py:161-162, 205-2` | b08ad1af1. Remaining: Fixed: RLTrainer replay entries now store detached log_prob/value; PrioritizedReplayBuffer uses (\|delta\|+1e-6)**alpha (signed/all-zero TD errors no longer crash); MuZero policy loss uses a clamped log (no NaN on underflow) and train_muzero returns [] on an empty dataset; MessageBus/CoordinationProtocol intent slots renamed to real NODE_ACTIONS types (same 6 slots); temporal_replay pads missing agents to the observed state width. Not done: the owner must decide whether the write-only MessageBus should feed agent observations (architecture change); the ray_distributed Learner item is PLAUSIBLE only (Ray not ins |
| spatial G-5: Graph mode fabricates result metadata. | low | `` | 92580270b. Remaining: Deferred: graph mode names the segmentation stage "segment" in stages_completed/stage_reports while imperative mode records "segmentation"; the graph name is pinned by tests/spatial_ai/orchestration/test_graph_pipeline_exports.py, so the owner must choose the canonical stage name for graph-mode results. |
| spatial I-5: Manifest validation is fail-open. | low | `` | e7ebf9729. Remaining: Deferred: unknown fields are still silently dropped by pydantic (extra="ignore") and lost on a write round trip; the owner must choose between preserving them (extra="allow", additive) and rejecting them (extra="forbid", rejects manifests that load today) for the v1.0.0 manifest contract. |
| spatial S-6: ArrayDigestCache keys on id(array). | low | `` | 3074090fd. Remaining: Deferred, owner decision: an array mutated in place keeps its cached digest, since detecting that needs a rehash that removes the N-3 memoization pinned by test_same_array_hits_cache; options are dropping memoization for writeable arrays or keeping the documented caller contract (do not mutate between lookups, or clear()/override). |
| tools T-16: rights policy default_owner never applies to v2 manifests; missing input mislabelled | low | `tools/apply_rights_policy.py:202, :223-225` | c008f28a4. Remaining: Deferred: policy default_owner never overrides a v2 entry owner of "UNSPECIFIED" (the builder default); treating "UNSPECIFIED"/"" as unset would change owner bytes in rights JSONL of runs that succeed today whenever a custom policy sets default_owner (the shipped policy/archive/rights_flags.yml uses UNSPECIFIED, so it is unaffected). Owner must decide whether the policy default_owner should take precedence over the manifest's UNSPECIFIED sentinel. |
| trust T-18: presence_security: the DCT watermark clamps the whole image to luma 16..235 and truncates payloads silently; dither is biased and alters alpha; LSB mo | low | `presence_security/watermarking.py:84, :73-76; presence_security/countermeasures.py:14-18; ` | 7b0a233e6. Remaining: Deferred to the owner (changes pixel bytes of runs that succeed today): whether DCT mode should stop clamping the whole luma plane to 16..235 and rewriting untouched blocks, and whether add_dither should round instead of truncate and leave alpha undithered. |

## Findings by subsystem

Status is one of fixed (with the repairing commits), partly fixed, deferred, rejected (did not reproduce or is
intended behaviour), not confirmed, or open. Paths are relative to `src/transformation_portal/` unless they
start with another top-level directory.

### FastAPI origin and portal helpers (app.py, portal/)

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| A1 | high | `portal/job_artifacts.py:766-771, 820-882; app.py:3650-3686; app.py:3794-3823` | data loss: When the output dir has no usable run card or batch manifest, the indexer walks the whole output_dir and records the first 200 files as this job's artifacts, whether or not the jo | fixed: e6157db09 |
| A2 | high | `portal/job_artifacts.py:542-605; run_card_*.json; app.py:7411-7423` | wrong job status: A job's run_summary, artifact list, state and error come from whichever run card is newest in output_dir at call time, not from the job's own run. Re-running into the same  | fixed: 871546239 |
| A3 | medium | `app.py:10939; app.py:3637-3638; orchestrator/job_execution.py:661-692` | DELETE races job finalisation | fixed: 10455df78 |
| A4 | medium | `app.py:6955; portal/asset_bundle.py:397; portal/asset_bundle.py:414` | unauthenticated 500 in the auth path | fixed: 9dccc8a30 |
| A5 | medium | `app.py:4388, 4420; app.py:4450; app.py:6402` | 500 from valid JSON: | fixed: 7ca1dc127 |
| A6 | medium | `app.py:9399-9439` | failed authentication is never rate-limited | fixed: d1f3155ee |
| A7 | medium | `app.py:7415-7423; app.py:7713-7714; app.py:10587-10588` | event loop blocked per request | fixed: 871546239 |
| A8 | medium | `portal/archive_index_preflight.py:116-129` | stale cache blocks dispatch | fixed: df18a8c36 |
| A9 | medium | `portal/archive_index_preflight.py:202-217; portal/path_security.py:366-376, 413-427` | O(rows × directory size): Each index row resolves each path component by listing the whole parent directory. The full scan runs inside POST /v1/jobs for fixity-scan. | fixed: df18a8c36 |
| A10 | medium | `app.py:3324-3354; app.py:3517-3529; R/r10_header_parse_cost.py` | A part-header block of any size up to the 1 MiB request limit is parsed by email.parser.BytesParser(policy=default) synchronously in the async handler. That parser is super-linear. For some  | fixed: e0761c27c |
| A11 | medium | `portal/job_artifacts.py:417-420; app.py:10587-10588; run_card_*.json` | one bad run-card file takes down the job list for everyone | fixed: a3d269081 |
| A12 | low | `app.py:9919-9953; ingest/upload_staging.py:221` | 500s from the staged-upload route | fixed: by an earlier fix in this audit. All three 500s (100k-deep and 5000-digit-int client_manifest, 300-char filename) now return the typed 400 INVALID_ARGUMENT envelope at this start point: the ingest fix |
| A13 | low | `app.py:6835-6845; app.py:2712-2718; app.py:3002-3008` | information disclosure: GET /v1/readiness needs no key and returns the absolute interpreter path, the absolute path of tools/archive_governance.py, the full runner argv, and the DA3 runtime  | fixed: 543753d65 |
| A14 | low | `app.py:11569-11594` | reclaim sweep overwrites a finished job | fixed: 8a0861af4 |
| A15 | low | `portal/asset_bundle.py:285-287` | Sixteen fingerprint lookups miss on every page load, and each miss re-reads and re-hashes the file on the event loop. The assets total 0.8 MB, so the cost is small. | fixed: 630087945 |

### Durable state: orchestrator/, storage/, events/, hardening/, api/

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| ORCH-01 | medium | `orchestrator/worker.py:305-315; orchestrator/job_execution.py:150-160, :450-451, :281-285, :302` | lease loss is recorded as a user-style canceled | fixed: 782cf2bb5 |
| ORCH-02 | medium | `orchestrator/storage/memory.py:121-134; orchestrator/storage/base.py:251-296; orchestrator/stor` | MemoryJobEventStore never purges deleted jobs | fixed: 2d1c5c944 |
| ORCH-03 | medium | `orchestrator/artifact_store/s3.py:677-705; app.py:10809; app.py:10871` | S3ArtifactStore.head() downloads the whole object on every artifact GET | fixed: ed62d5803 |
| STOR-01 | medium | `storage/cas_replication.py:363-364; cas_store.py` | CASReplicator.pull_many writes the final CAS path non-atomically, then never repairs it | fixed: 37ee55081 |
| STOR-02 | medium | `storage/cas_replication.py:197` | push_many treats a short or missing present list as "peer has everything" | fixed: 304069864 |
| STOR-03 | medium | `storage/dataset_cas.py:352; app.py` | DatasetCAS.materialize path traversal via manifest logical_path | fixed: 9662dd0b7 |
| ORCH-04 | low | `orchestrator/job_execution.py:667-669; app.py` | post-run indexing failure leaves a job that is never finished | fixed: b42504d8e |
| ORCH-05 | low | `orchestrator/artifact_store/local.py:487; orchestrator/artifact_store/s3.py:794; orchestrator/a` | LocalArtifactStore.list_for_job is not relative_path-sorted | fixed: 22ca30af5 |
| ORCH-06 | low | `orchestrator/artifact_store/s3.py:926` | S3 bulk delete(job_id) leaks raw botocore exceptions | fixed: 22ca30af5 |
| STOR-04 | low | `storage/dataset_cas.py:179-184; a_b_1.json` | DatasetCAS._manifest_path collisions silently replace another dataset | fixed: 9ce41b780 |
| STOR-05 | low | `storage/cas_store.py:775-781` | ArtifactStore.gc() deletes other writers' in-flight temp files | fixed: 01f883567 |
| STOR-06 | low | `storage/cas_store.py:738-748` | ArtifactStore.materialize is check-then-act on the destination | fixed: d781b5ca7 |
| STOR-07 | low | `storage/cas_store.py:322-357` | open_verified relabels the caller's OSError as a CAS failure | fixed: 166ef915d |
| EVT-01 | low | `events/store.py:158; *.json; 2023-11-14/evt-0.json` | EventStore.clear() leaves persisted events; limit=0 returns everything | fixed: 16b4b5f48 |

### Lux Depth V3 orchestrator core

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| F1 | high | `orchestrator.py:4030-4041; orchestrator.py:4088; pipeline_coordinator.py:284-294` | one runtime fallback re-homes the rest of the batch on the fallback backend; on the production path the whole batch then fails at final evidence | fixed: 6d2f5fa69 |
| F2 | high | `orchestrator.py:4903; orchestrator.py:4959; orchestrator.py:5696` | output_key.stem truncates at the first dot, so PBR maps, mask artifacts and the Materials hand-off of different inputs collide | fixed: 59a701889 |
| F3 | medium-high | `orchestrator.py:4694-4703; config.py:738-747` | the documented APEX v2-auto recovery is unreachable | deferred: docs-versus-code contract question (APEX v2-auto recovery) |
| F4 | medium | `orchestrator.py:6302; orchestrator.py:797-799; *_combined_provenance.json` | the provenance sidecar records the git SHA of the caller's working directory | fixed: 816f8b228 |
| F5 | medium | `orchestrator.py:6267, :6295-6313, :6326-6340; depth/render_0_tif_c7f9fff8_depth_metadata.json` | the exiftool requirement for TIFF/RAW is only discovered in _write_manifest, after full inference of every image | fixed: 70a6f2d80 |
| F15 | medium | `probe_prepared_hash_amplification.py; probe_prepared_hash_breakdown.py` | on the production path every output byte is read and SHA-256-hashed about 10× per fresh batch and about 14× per resume batch | partly fixed: 46e3d195d. Remaining: Deferred to the execution_evidence owner: the preliminary evidence build in _activate_carrier_outcome_projection (2.0x), build+verify in the evidence writer (4.0x) and reuse prim |
| F6 | medium | `orchestrator.py:8057-8063, :3526-3534` | prepared runs build the full reuse snapshot twice per image | fixed: 9b9fb6599 |
| F7 | medium | `input_discovery.py:187; input_discovery.py:198-203; orchestrator.py:8316-8321` | discovery filters match the ancestors of the input root | see other row: handled by the lux-evidence cluster (F-03) |
| F8 | medium | `artifact_manager.py:248; orchestrator.py:6916-6929` | a depth-reuse run that re-executes V2 feeds V2 the un-enhanced source | fixed: 0417cc635 |
| F9 | medium | `orchestrator.py:4717-4732` | v2-auto after total depth failure writes a manifest claiming backend success | partly fixed: 224715af9. Remaining: Deferred: result["backend"]/fallback_used and the run-card backend_summary still carry the start-up selection, because validators/run_card_validator.validate_run_card_backend_sem |
| F10 | medium | `orchestrator.py:4546-4548` | a stale float-depth .npy is indexed in the next run card and preferred for PBR regeneration | fixed: 1c9be5de0 |
| F14 | medium | `orchestrator.py:4521-4528, :4547, :4681` | Stage-A artifacts are overwritten in place before the manifest that describes them | fixed: 67c53c533 |
| F11 | medium-low | `config_resolver.py:864-896; manifest.py:372-399; orchestrator.py:755` | the Stage-A reuse fingerprint omits settings that change depth | fixed: 7cebe9277 |
| F12 | medium-low | `orchestrator.py:9775-9783` | one unreadable input aborts the whole batch at run-card assembly | fixed: 579ea77a9 |
| F13 | low-medium | `orchestrator.py:3741-3760` | verify_outputs never checks the V2 deliverable | fixed: d8491c6c3 |
| F16 | low | `config.py:734-747; execution_lifecycle.py:1519-1523` | EnhanceConfig.__post_init__ is not idempotent for apex-strict | fixed: 2b2adcf60 |
| F17 | low | `orchestrator.py:8011-8015` | parallel-preprocess error rows carry image_input instead of image | fixed: 886dd79e7 |

### Lux Depth V3 evidence, lifecycle, cache, manifest, I/O, CLI

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| F-01 | high | `execution_lifecycle.py:185; orchestrator.py:8325; /e2e_order_detail.py` | prepared runs fail at the end when plan order differs from sorted(Path) order | fixed: 45860e09e |
| F-02 | medium |  | confined_relative_path resolves root-relative paths against the process CWD first | fixed: b116885b1 |
| F-03 | medium | `input_discovery.py:158-213; input_discovery.py:44-56; __main__.py:1177-1182` | input-discovery filters apply to the input root's own path, depend on spelling, and miss two output folders | fixed: 40838d6a4 |
| F-04 | medium | `depth_cache.py:1972-1990; depth_cache.py:2056-2132; depth_cache.py:1676-1716` | depth cache at capacity re-verifies every cached object on every store() | fixed: b1f95e1fc |
| F-05 | medium | `provenance.py:559-667; provenance.py:130; provenance.py:627` | capture_provenance re-probes invariant tools and re-hashes the input for every image | partly fixed: 1a8f98f44. Remaining: Deferred to the orchestrator.py owner: per-image git rev-parse (a process-wide memo would let the sidecar diverge from the per-orchestrator manifest v3_git_revision pinned by tes |
| F-06 | low-medium | `validators/run_card_integrity.py:583-584; run_card_contract.py:23-28; validators/run_card_integ` | verify_run_card_integrity and assess_run_card_release raise on malformed run cards | fixed: ac023e9b4 |
| F-10 | low-medium | `reconstruction_runner.py:56-77; reconstruction_runner.py:394-397; spatial_ai/reconstruction/gau` | run_scene_reconstruction (prepared path) rescales the caller's "immutable" SceneContext cameras in place | fixed: a047281c5 |
| F-07 | low | `execution_lifecycle.py:144-187` | plan accepts input names the evidence schema rejects; the run fails only after all work is done | fixed: 9ceb539fd |
| F-08 | low | `io_atomic.py:313-399; io_atomic.py:470-502; io_atomic.py:505-569` | io_atomic: only atomic_write_bytes is durable; atomic_write_with_fd double-closes and leaks | fixed: 17f2b46f5 |
| F-09 | low | `__main__.py:171-176; __main__.py:792-838; __main__.py:1073-1141` | CLI option validation gaps | fixed: 93e483c61 |
| F-11 | low |  | carrier outcome-projection validation is O(inputs × outcomes) | fixed: b2527c1d1 |
| F-12 | low | `manifest.py:478-479; provenance.py:495; provenance.py:493` | manifests and provenance use the locale default encoding for UTF-8 JSON | fixed: 7463c1c9c |
| F-13 | low | `security.py:144-149; reconstruction_manifest.py:82-85; reconstruction_runner.py:80-83` | "nonlossy" path sanitiser truncates to 255 chars and callers append suffixes | fixed: 30f084797 |
| F-14 | low | `run_card_integrity.py:554-556` | reconstruction diagnostics report percentiles that are not measured | deferred: Per-camera reprojection_rmse/max/p50/p95/p99 in tp.reconstruction_diagnostics.v1 are the global scene RMSE by construction (np.full(points_observed, scene.rmse)); Scene3D carries no per-camera 2D obse |

### Lux Depth V3 pixel, inference, segmentation, materials, PBR

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| LC-01 | high | `lux_depth_v3/segmentation/efficient_sam.py:751-763; lux_depth_v3/segmentation/_cache.py:112-121` | dead acceptance gate in CLIP material labelling | deferred: enabling the CLIP gate needs a calibrated threshold and changes labelling for every run |
| LC-02 | medium | `lux_depth_v3/segmentation/_cache.py:253-322; lux_depth_v3/segmentation/registry.py:300-374; lux` | segmentation cache reuses degraded fallback results | fixed: f86141e0a |
| LC-03 | medium |  | 16-bit / float grayscale inputs clipped before depth inference | fixed: e4cd565fe |
| LC-04 | medium | `lux_depth_v3/v2_enhance.py:569-598, :698, :775-782, :1082-1097, :1295, :408-426, :1403-1413` | 16-bit PNG bit depth misdetected in V2; wrong evidence, one crash | fixed: a864f4a09 |
| LC-05 | medium | `lux_depth_v3/inference.py:578-587` | pinned model revision dropped on TypeError | fixed: ca1a216c3 |
| LC-06 | medium | `lux_depth_v3/segmentation/sam_vit_h.py:304-335; lux_depth_v3/segmentation/registry.py:423-427; ` | SAM ViT-H reports mask IoU as material confidence | fixed: 71a0fc5a1 |
| LC-07 | medium | `t_square.py` | float truncation in aspect-preserving resize costs a 14-px band | deferred: changes depth output geometry for ordinary inputs |
| LC-08 | medium | `lux_depth_v3/scene_groups.py:325-339` | parent_dir scene grouping splits one directory | fixed: 16e5120ab |
| LC-09 | medium | `lux_depth_v3/pbr.py:62-78, :156-168, :379-400, :437-446, :477-486; t_pbr.py` | PBR numpy path and batched torch path disagree | partly fixed: 4df743481. Remaining: Deferred: the numpy reference itself computes N=(+dx,+dy,1) against the documented (-dx,-dy,1) and truncates instead of rounding (flat normal 127 not 128); fixing either changes  |
| LC-10 | medium | `lux_depth_v3/depth_writer.py:86-104; lux_depth_v3/orchestrator.py:4516-4527` | metric depth clipped by a value-range heuristic | fixed: 49adfb5a8 |
| LC-11 | medium | `lux_depth_v3/pbr_cli.py:101-103, :284-291, :319-349; lux_depth_v3/pbr_processor.py:130-134; _de` | pbr_cli batch mode processes each scene 2–3 times | fixed: a384e12b5 |
| LC-12 | medium | `lux_depth_v3/segmentation/efficient_sam.py:604-633` | full-image pass per connected component | fixed: f868fe7a3 |
| LC-13 | low | `lux_depth_v3/pbr_processor.py:82-92, :157-166; lux_depth_v3/pbr_writer.py:79-86` | partial PBR write failure reported as success | fixed: ac2842b93 |
| LC-14 | low | `lux_depth_v3/v2_enhance.py:526-546` | find_depth_map treats the image stem as a glob | fixed: f9e744c52 |
| LC-15 | low | `lux_depth_v3/depth_writer.py:188-209, :61-66` | depth write statistics include zero sentinels | deferred: Reproduced: atomic_write_depth_u16_png_with_stats computes min/mean/std after unsupported and non-finite samples become 0 (stats.min=0.0 vs normalization.source_min=0.36). Deferred because the value r |
| LC-16 | low | `lux_depth_v3/depth_writer.py:355-360` | read_depth_u16_png scales differently by decoder | fixed: cb7ae2619 |
| LC-17 | low | `lux_depth_v3/pixel_ops_executor.py:346; lux_depth_v3/pixel_ops_registry.py:46-48, :75` | pixel ops: dead invalid_mask_shape guard, broken grayscale branch, alpha edited | fixed: 524932908 |
| LC-18 | low |  | normalize_exif_orientation copy branch is unreachable | fixed: 5ebbc2db7 |
| LC-19 | low | `lux_depth_v3/scene_integrity.py:457-459; lux_depth_v3/validators/run_card_integrity.py:493` | incomplete scene validation | fixed: 11863ae2a |
| LC-20 | low | `lux_depth_v3/linear_verify.py:215-228` | gamma-detection heuristic is inverted | fixed: 5029fceeb |
| LC-21 | low | `lux_depth_v3/protocols/depth_model.py:449-476` | DepthModelRegistry role lookup matches by model_id | fixed: 485027a6e |

### Lux Depth V4/V5/V6 and materials_v4

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| LN-1 | medium | `lux_depth_v4/photography.py:168; input_preflight.py; /png_chunks_probe.py` | Valid large PNGs are rejected by a hard 8192-chunk cap | fixed: 5837091d7 |
| LN-2 | medium | `depth_pro.py:486-495` | Standalone V6 writes completion evidence its own verifier cannot read once the batch is large | fixed: 745d3e6de |
| LN-3 | medium | `materials_v4/__main__.py:192-193; evaluation.py:24-25` | materials-v4 evaluate reads 16 MiB but rejects anything over 1 MiB | fixed: d69ee0e45 |
| LN-5 | medium |  | lux-depth-v6 --verify runs the complete V5 parent replay twice | fixed: 18d31f42a |
| LN-7 | low |  | fit_calibration reloads the whole evidence bundle and source photograph for every record | fixed: 84d5254d8 |
| LN-4 | low | `lux_depth_v4/pipeline.py:531` | V6 re-parses and re-validates the whole V5 plan and evidence per image (O(N²)) | fixed: 40be1d1a8 |
| LN-9 | low | `lux_depth_v6/depth_pro.py:262-270` | Depth Pro default limits cannot admit a 24 MP photograph, and the error blames max_pixels | fixed: 413137a23 |
| LN-10 | low | `lux_depth_v4/backend.py:202; lux_depth_v4/worker.py:148; lux_depth_v5/worker.py:164` | Worker RPC frame is ASCII-escaped, but worker ingress applies the 1 MiB plan decoder | fixed: 4c475afae |
| LN-11 | low | `lux_depth_v4/backend.py:45-65; raw.py:96` | DA3 and RAW workers inherit the parent's full environment | partly fixed: 976506eb4. Remaining: Deferred: a V6-style allow-list that also drops loader-injection (LD_PRELOAD, DYLD_*) and behaviour switches (PYTORCH_ENABLE_MPS_FALLBACK, HF_ENDPOINT); the owner must decide whi |
| LN-8 | low | `lux_depth_v6/source.py:51; plan.py:153-157; plan.py:299-312` | V6 accepts max_pixels up to 200 M but GradePlan hard-codes 100 M | fixed: ff5f89600 |
| LN-6 | low | `materials_v4/evaluation.py:266` | Calibration (ECE) bins put boundary confidences in the bin below | fixed: db875efe9 |

### core/, determinism/, execution_graph/, stage_graph/

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| CORE-01 | high | `core/execution_wrapper.py:440-468; core/execution_identity.py:553` | cache identity collision: input ids are appended without their key names and then sorted, so the identity is an unordered bag of content hashes. The generic fallback hashes str(value): numpy | fixed: 8012e5f01, 16de62e5a |
| CORE-02 | high | `core/execution_wrapper.py:205-249; lux_depth_v4/pipeline.py:252` | A waiter deletes a live lock once its timestamp is older than 2 × the waiter's own timeout. | fixed: 3f2d732dd |
| CORE-03 | high | `core/security/task_signing.py:284` | fail-open on missing key: with no authorised keys the authorisation step is skipped, and the signature is checked against the public key carried inside the task. | fixed: 91f4b1379 |
| CORE-04 | medium | `core/security/task_signing.py:36-51` | canonicalisation before signing | fixed: 19c4fb131 |
| CORE-05 | medium | `core/security/signing.py:298-302` | the signature covers only manifest_hash + root_hash. issuer, signed_at, metadata and certificate_id are unauthenticated. verify() — the call the module docstring recommends — trusts the key  | deferred: Reproduced at 1208cab: core/security/signing.py signs only (manifest_hash + root_hash), so issuer/signed_at/metadata/certificate_id can be rewritten and still verify, and CertificateVerifier.verify()  |
| CORE-06 | medium | `core/security/tls.py:186-198` | availability: the TLS handshake runs inside the accept loop with no timeout, including for ThreadedTLSServer. One client that connects and sends nothing blocks all others. | fixed: 23c31edf3 |
| CORE-07 | medium | `core/security/fs_guard.py:311-315, :330-333; runtime/engine.py; runtime/sandbox_executor.py` | "atomic" writes stage through one fixed <name>.tmp per target, so concurrent writers rename each other's staging file. There is also no fsync. | fixed: 38df35af8 |
| CORE-08 | medium | `core/_cas_helpers.py:161-181; core/cas_dag_executor.py:488-493` | cached results do not round-trip | fixed: dbe70f805 |
| CORE-09 | medium | `core/cas_dag_executor.py:730-731` | false assurance: it compares identity.cas_id across runs — the input identity — not output hashes. The last stage in execution order is never checked. | fixed: e6af13d89 |
| CORE-10 | medium | `determinism/cli.py:234-271` | fail-open verification: the tensor is loaded by directory name and never re-hashed against the artifact id. The .npy is preferred although the id is defined over the .bin bytes. | fixed: 4d03da442 |
| CORE-11 | medium | `determinism/jcs.py:52` | canonicalisation: np.float64 is a float subclass, and under NumPy 2 its repr is np.float64(…). Non-string keys fail in the sort key before the intended TypeError. | fixed: bde6221f5 |
| CORE-12 | medium | `core/platform_matrix.py:341-379; core/execution_identity.py:559-560` | it runs ["pip", "freeze"] — the first pip on PATH, not the running interpreter's — on every call with no caching. It is called once per stage for identity and again per cache hit. | fixed: 16afccfe4, fbd595c30 |
| CORE-13 | medium | `core/execution_identity.py:265` | stale cache after code edits | fixed: c02219e9e |
| CORE-14 | medium | `core/execution_identity_v4.py:42-58; core/execution_identity_v5.py:26,48; core/execution_plan_v` | accidental O(N²): from_plan re-parses and fully re-validates the plan from bytes, then decodes it twice more. lux_depth_v4/pipeline.py:531 calls it once per stage per input, and each call is | fixed: 00becd803 |
| CORE-15 | medium | `stage_graph/stage.py:107-109; stage_graph/graph.py:235-240; core/cas_dag_executor.py:498` | skip logic: StageStatus.SKIPPED is treated as failure, although UpscalingStage(scale_factor=1.0) returns SKIPPED with a valid pass-through artifact. | fixed: 65d04e743 |
| CORE-16 | medium | `stage_graph/stages/upscaling.py:233` | cache key collision: Evidence: CONFIRMED — $R/stage_graph_checks.py: [3] scale 1.25 -> (50, 50, 3) \| then scale 1.2 -> (50, 50, 3) cache_hit = True (expected (48, 48, 3)) reported scale = 1 | fixed: ff465a80f |
| CORE-17 | medium | `stage_graph/stages/depth_pro.py:230-238; stage_graph/stage.py:295-337; depth/backends/depth_pro` | legacy stage cache, Depth Pro | fixed: 16928e9d4, bb3d70772 |
| CORE-18 | low | `core/security/task_signing.py:336` | any clock skew rejects a fresh task. Nonce eviction pops an arbitrary entry, so an evicted, still-fresh task replays. | fixed: a3ca0689c |
| CORE-19 | low | `core/security/signing.py:154` | key handling: Evidence: CONFIRMED — $R/security_signing.py: [6] private key file mode: 0o644 | fixed: de81ab692 |
| CORE-20 | low | `core/cas_dag_executor.py:674-675` | Evidence: CONFIRMED — $R/dag_cache_fidelity.py: [4] invalidate(before=datetime.now()) -> TypeError can't compare offset-naive and offset-aware datetimes | fixed: 435c674f5 |
| CORE-21 | low | `core/cas_dag_executor.py:525-526` | the cache root is opened with O_NOFOLLOW, so a symlinked cache_dir stores entries but silently never reads them. | fixed: c03c816d5 |
| CORE-22 | low | `stage_graph/graph.py:107-113` | partial failure leaves state | fixed: 1f6122e50 |
| CORE-23 | low | `stage_graph/policy.py:186` | "upscale" is not a substring of the real stage name "upscaling". | fixed: a2153ba59 |
| CORE-24 | low | `execution_graph/distributed_executor.py:192; execution_graph/scheduler.py:324-352` | error propagation: Evidence: CONFIRMED with a stub ray module — $R/stage_graph_checks.py | fixed: f15d0e934 |
| CORE-25 | low | `execution_graph/patcher.py:125-126, :141-142` | the "old" value is read after the flag is overwritten. | fixed: 2f046f480 |
| CORE-26 | low | `core/photographic_export.py:35-36; photographic_delivery/exports.py:50; photographic_delivery/n` | off-by-one: int(width * (long_edge / max_side)) truncates. Used by photographic_delivery/exports.py:50. | deferred: resize_long_edge feeds hash-pinned photographic export deliverables |
| CORE-27 | low | `core/config/preset_health.py:217-218, :230-231, :320-322` | crash on reachable input: Evidence: CONFIRMED — $R/misc_checks.py | fixed: 5dd27fc5b |
| CORE-28 | low | `core/ml_dependency_health.py:34; core/geometry/camera_params.py:61-64; core/security/sanitizati` | validation gaps: Evidence: CONFIRMED — $R/misc_checks.py | fixed: bedfc9619 |
| CORE-29 | low | `core/storage/export_manager.py:145-146; core/storage/autotune_helpers.py:43; core/validation/me` | helpers documented as having no production imports | fixed: 173ad05ae |
| CORE-30 | low/medium | `core/processing/tiling.py:189-203` | seam maths: Reasoning: the default gaussian window subtracts only the corner minimum, so a tile's weight is still about 28% of the blend at its own edge and then drops to zero one pixel late | fixed: 49d85f543 |
| CORE-31 | medium | `stage_graph/stages/materials.py:237-244, :299-303` | Reasoning: skimage.color.rgb2hsv returns hue in [0, 1], but the wood mask tests hue > 10 and hue < 40, so it would always be empty. scikit-image is not installed, so this was not run. | not confirmed: plausible only |
| CORE-32 | medium | `stage_graph/stages/depth.py:124-128` | Reasoning: _load_model hard-codes Depth-Anything-V2-Small-hf regardless of model_size, which still goes into the cache key and metadata. It maps every device except "cuda" to CPU, so MPS is  | not confirmed: plausible only |

### Trust surfaces (tp/, attestation/, plugins/, compliance/, presence_security/) and runtime/

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| T-01 | high | `attestation/gpg.py:140-158; tools/verify_run_card_attestation.py:175-179; lux_depth_v3/validato` | GPG attestation verification accepts revoked and expired keys; the DSSE path never binds the signer | fixed: 5529b9138 |
| T-02 | high | `plugins/registry.py:166-209, :223-242; *.py` | PluginRegistry.discover_plugins bypasses the external-plugin opt-in and the ADR-049 trust store | fixed: 27915c078 |
| T-03 | high | `runtime/sandbox.py:310-316, :331-336; core/security/tenant.py:450-462` | Sandbox workspace confinement is lexical only | fixed: 5aef10fff |
| T-04 | high | `src/tp/crypto/ct_merkle.py:488-497; lux_depth_v3/artifact_tree.py` | tp.crypto.verify_ct_consistency_proof rejects about a quarter of honest proofs and does not bind the proof to the tree sizes | fixed: c84fc7f0b |
| T-05 | high (latent) | `runtime/process_executor.py:243` | ProcessExecutor.run joins the child before draining the result queue | fixed: 0d59c3d98 |
| T-06 | high (latent) | `runtime/ledger.py:214` | ImmutableLedger: a failed append bricks verification, concurrent handles fork the chain, and tampering verifies | partly fixed: 6764797e9. Remaining: Deferred for the owner (ADR-051 already marks ImmutableLedger frozen): the chain is still unkeyed and the signer still does not sign entry hashes, so a writer holding the issuer  |
| T-07 | medium | `plugins/signing.py:28-37, :70-93; plugin.json; plugins/loader.py:375-388, :434-442` | a valid ADR-049 manifest signature authorises arbitrary code | deferred: binding code digests into the ADR-049 manifest is a contract change |
| T-19 | medium | `compliance/licensing.py:141-146; config/presets/apex_research.yaml; config/presets/experimental` | validate_non_commercial_preset only inspects top-level model.hf_id | fixed: d4e80eb55 |
| T-08 | medium (latent) | `runtime/certificate.py:192-194, :264-265, :327-328; runtime/execution_manifest.py:128-147; mani` | ReproducibilityCertificate authenticates only manifest_hash:root_hash | deferred: what the reproducibility certificate authenticates is a signing redesign |
| T-09 | medium (latent) | `runtime/engine.py:319-329; runtime/replay.py:212-213` | ExecutionEngine hashes wall-clock data into provenance hashes; ReplayEngine cannot replay engine-produced nodes | fixed: 27029a1eb |
| T-10 | medium (latent) | `runtime/cas_server.py:174-176` | CAS replication server writes objects non-atomically and never repairs them; CASServerProcess cannot start under spawn | fixed: b38a29ac3 |
| T-11 | medium | `plugins/loader.py:434-442; signing.py:92; registry.py:192-195` | PluginLoader: module-name collisions, one bad manifest aborts discovery, builtin plugins undiscoverable, wrong dependency check, "invalid" plugins still execute | fixed: d43eb2446 |
| T-12 | medium (latent) | `runtime/scheduler.py:202, :209` | ClusterScheduler: priority=0 and max_retries=0 are replaced by the defaults; equal priorities are not FIFO | fixed: 44c9daa11 |
| T-13 | medium (latent) | `runtime/gpu_session_pool.py:331` | GPUSessionPool: stale results from a failed map() are returned as the next batch's results; unpicklable tasks vanish | fixed: 8ee1197c6 |
| T-14 | medium (latent) | `runtime/engine.py:257-263; runtime/gpu_pool.py:164; runtime/gpu_semaphore.py:206-213` | GPU accounting holes: Evidence: CONFIRMED — repro_runtime_scheduling.py and repro_gpu_session_pool.py | fixed: ee7ff6dbe |
| T-15 | medium (latent) | `runtime/autoscaler.py:263-267` | GPUAutoscaler cannot scale up from zero or up to min_workers; its backlog rule is unreachable | fixed: 6ee638ac9 |
| T-17 | medium (latent) | `runtime/gpu_model_cache.py:143-160` | ModelCache.get_or_load loads once per waiting thread; max_memory_bytes ignores the entry being inserted; max_entries=0 crashes | fixed: 1508ff0da |
| T-16 | low (latent) | `runtime/autoscheduler.py:209, :427, :460-465` | ClusterAutoscheduler: heartbeat un-drains a worker; default dispatch cannot serialise a ProcessTask; scale_check never fires under load | fixed: 8a47f8e72 |
| T-18 | low | `presence_security/watermarking.py:84, :73-76; presence_security/countermeasures.py:14-18; prese` | presence_security: the DCT watermark clamps the whole image to luma 16..235 and truncates payloads silently; dither is biased and alters alpha; LSB mode writes lossy files | partly fixed: 7b0a233e6. Remaining: Deferred to the owner (changes pixel bytes of runs that succeed today): whether DCT mode should stop clamping the whole luma plane to 16..235 and rewriting untouched blocks, and  |
| T-20 | low | `src/tp/phase4/canonicalize_capture_metadata.py:324-334; tools/extract_capture_metadata.py:76-90` | Phase 4C extraction: hostile EXIF values raise raw exceptions instead of WARN_* codes | fixed: aab8f6612 |
| T-21 | low | `src/tp/phase4/verify_phase4_chain.py:417, :424-483; src/tp/phase4/validation_helpers.py:221-229` | Phase 4F failure-report inputs are not sanitised | fixed: d3ef56008 |
| T-22 | low | `plugins/builtin/processors.py:292-294` | builtin ResizeProcessor: the "fit within bounds" branch is unreachable | fixed: 009c0abbe |
| T-23 | low | `attestation/dsse.py:18` | DSSE pre_auth_encode writes the character count, not the UTF-8 byte count | fixed: 9627f77f7 |
| P-1 | low | `attestation/sigstore.py:57-59` | cosign 2.x refuses; cosign 1.x would accept any Fulcio identity. cosign is not installed. | not confirmed: plausible only |
| P-2 | low | `src/tp/phase4/canonicalize_capture_metadata.py:459-461; tools/extract_capture_metadata.py:55` | I confirmed that with --input-root . a file -overwrite_original.dng becomes the argv element '-overwrite_original.dng'; exiftool's handling is untestable here. | not confirmed: plausible only |
| P-3 | low | `runtime/cuda_ipc.py:74-90` | GPU and torch are required. | not confirmed: plausible only |

### Governed CLIs and repository scripts (tools/, scripts/)

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| T-01 | high | `merkle_roots.json` | the signed Merkle roots are never tied back to the hash manifest | fixed: 58c1781cc |
| T-02 | high | `tools/archive_bagit.py:342-452` | BagIt validator passes bags that are not valid | fixed: 8e3404e94 |
| T-03 | high | `scripts/validation/verify_checkpoint_sha256.sh:53-58` | checkpoint checksum verifier cannot fail | fixed: 5e629aed4 |
| T-11 | medium | `tools/regulatory_export.py:1397-1422; //attacker.example/verify.sh` | governance-export verification ignores part of the export | partly fixed: 69a865936. Remaining: Deferred: without that flag a rewritten governance_profile_id still verifies (the export is the only source of the id and the documented verify command does not pass it); the own |
| T-12 | medium | `tools/verify_run_card_attestation.py:127-200` | run-card attestation verifier passes with nothing verified | fixed: a9289092f |
| T-13 | medium | `tools/build_archive_manifest_v2.py:100-107; tests/test_archive_hash_integrity.py` | manifest v2 builder silently drops data rows that start with # | fixed: 2dc2f6a78 |
| T-14 | medium | `tools/verify_hash_manifest.py:375-408` | fixity verify "passes" with zero rows and a nonexistent archive root | fixed: 29c5f6a56 |
| T-04 | medium | `tools/manifest_telemetry.py:288-307; .github/workflows/build.yml:1236-1246; manifest_merkle.jso` | Merkle proof verifier has no anchor | fixed: bfdfa6817 |
| T-05 | medium | `scripts/security/verify_banned_dependencies.py:101-128` | banned-dependency gate blind spots | fixed: 0f58abdd8 |
| T-06 | medium | `scripts/validation/scan_todo_inventory.py:240-251; depth/models/depth_anything_v2.py:369` | TODO governance gate passes vacuously | fixed: f991190af, 12be86fbf |
| T-07 | medium | `scripts/ci/lint_runner.sh:136-151; .github/workflows/quality-gate.yml:37` | lint runner: advisory mode blocks, and pylint is skipped after a flake8 failure | fixed: 48396bd1a |
| T-08 | medium | `tools/performance_ledger.py:391-417, :433-435, :733-735` | performance ledger --strict always fails; failure-rate branch is unreachable | fixed: 518233a35 |
| T-09 | medium | `scripts/setup/install_depth_pro_runtime.sh:79-82, :142; install_da3_runtime.sh:156-216` | Depth Pro installer wipes whatever --venv-dir points at | fixed: 3c516b233 |
| T-10 | low | `scripts/ci/track_test_flakes.py:30-67, :212-237; scripts/ci/analyze_flakes.py:21-27, :168-178` | --ledger is parsed but ignored | fixed: d3d563349 |
| T-15 | low | `tools/export_prov_stac.py:133-134, :156-162, :169-173, :259` | STAC export writes catalog links that do not resolve | fixed: 6c649d8f8 |
| T-16 | low | `tools/apply_rights_policy.py:202, :223-225` | rights policy default_owner never applies to v2 manifests; missing input mislabelled | partly fixed: c008f28a4. Remaining: Deferred: policy default_owner never overrides a v2 entry owner of "UNSPECIFIED" (the builder default); treating "UNSPECIFIED"/"" as unset would change owner bytes in rights JSON |
| T-17 | low | `tools/bundle_root_common.py:172-173` | malformed bundle manifest gives a traceback instead of the documented exit code | fixed: c1de28580 |
| T-18 | low | `tools/parse_machine_json.py:64-68` | reference machine-JSON parser exits 0 on an error envelope | fixed: a9cadb89f |
| T-19 | low | `scripts/validation/check_raw_json_usage.py:40-46; scripts/ci/check_no_tautological_tests.py:119` | two AST gates with easy false passes | fixed: f6f15800e, 01f22add9 |
| T-20 | low | `scripts/setup/install_fastvlm_runtime.sh:15,122; scripts/setup/pre-commit-check.sh:55,147; scri` | reasoning only; no bash 3.2 or multi-user host here | not confirmed: plausible only (needs bash 3.2 / multi-user host) |

### Managed front door (web/secure-landing) and Cloudflare worker

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| F-01 | high | `cloudflare/transformationportal-worker/src/index.ts:44-46, 77-79, 99-113; web/secure-landing/li` | Worker and frontdoor disagree about the public host | deferred: plausible integration finding that depends on deployed Cloudflare/Vercel configuration |
| F-02 | medium | `web/secure-landing/app/login/route.js:464-489; web/secure-landing/lib/sessions.js:64-71, 230-25` | login throttle is check-then-act and keyed on a client-supplied address | fixed: 08b027e14 |
| F-03 | medium | `web/secure-landing/lib/session-store/sqlite-store.js:78-81; lib/sessions.js:95-110; app/login/r` | SQLite store never deletes expired sessions | fixed: 0c769e38e |
| F-04 | medium | `web/secure-landing/app/v1/portal/rum/route.js:96-116, 137-157` | unauthenticated POST /v1/portal/rum buffers and re-sends unbounded bodies with the backend API key | fixed: 95eab7f51 |
| F-05 | medium | `web/secure-landing/portal-src/portal.template.js:11010-11017` | live log pane does a full-text rewrite and forced reflow per log line, and is never trimmed | fixed: 6b120931b |
| F-06 | low-medium | `web/secure-landing/lib/proxy.js:4-23, 49-53; /route.js:315-369` | /v1 proxy forwards hop-by-hop request headers; such requests always get 502 | fixed: 9ddd8a72f |
| F-07 | low-medium | `web/secure-landing/scripts/seed-frontdoor-user.mjs:40, 84-90` | seed-frontdoor-user.mjs truncates --flag=value at the second = and trims passwords | fixed: 7d48f550f |
| F-08 | low-medium | `web/secure-landing/portal-src/portal.template.js:4879-4934, 5040-5053, 5080-5086, 5663-5670` | portal bootstrap retry counter is never reset | fixed: 296203da2 |
| F-09 | low | `web/secure-landing/lib/sessions.js:200-207; app/login/route.js:433; app/logout/route.js:56` | CSRF comparison throws on multi-byte tokens, giving HTTP 500 instead of 403/303 | fixed: c0da2d778 |
| F-10 | low | `web/secure-landing/app/logout/route.js:47-53` | /logout CSRF-failure path still clears the cookie and leaves the server session alive | fixed: d3cb141fc |
| F-11 | low | `web/secure-landing/portal-src/portal.template.js:12286-12295, 12328-12332, 12907-12911` | unguarded localStorage in init() aborts portal start-up when storage is denied | fixed: 2db1496c5 |

### Depth backends and depth tooling

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| D-12 | high | `depth/tools.py:381; pipelines/depth_tools.py` | depth tools batch driver pairs depth maps with the wrong photograph, overwrites outputs, reports 0 errors | fixed: f40afde3f |
| D-01 | high | `depth/pipeline.py:481-518; examples/batch_process.py:62` | ArchitecturalDepthPipeline.batch_process(parallel=True) (the default) pickles the whole pipeline per image | fixed: 505679698 |
| D-22 | high | `depth/models/depth_anything_v2.py:579-580; depth/depth_processor.py:98-99; depth_intelligence/d` | PLAUSIBLE — DA2 depth orientation is inverted relative to how it is documented and consumed | deferred: plausible only, needs model weights to confirm DA2 orientation |
| D-15 | medium | `depth/backends/da3.py:1111-1125; depth/backends/synthetic.py:101; depth/backends/depthcrafter.p` | ndarray inputs to DepthBackend.compute() are mangled, differently per backend | partly fixed: ee387268f. Remaining: Deferred: a float frame with max > 1.0 is still read as [0,255] pixels in every backend (documented DA2/synthetic/DepthCrafter convention, Lux V3 inference instead clips to [0,1] |
| D-23 | medium | `depth/backends/da3.py:1338-1348,1367-1370; depth/backends/depth_pro.py:537-540` | DA3 and Depth Pro parents accept an exit-0 worker's depth without validating it; malformed outputs escape the failure classification | partly fixed: 56fc75e66. Remaining: Deferred: depth shape equality with the echoed input_size, because the legacy DA3 engine returns depth at the model's processing grid (default 504 long edge), so the owner must d |
| D-17 | medium | `depth/backends/depth_pro.py:339-348; da3.py:483-496; depth/backends/depth_pro_worker.py:38-85` | Depth Pro subprocess calls have no timeout and inherit the parent's Python environment | partly fixed: ec09d9c8f. Remaining: Deferred: installed-package layouts (no checkout src) still forward the caller's PYTHONPATH because test_license_enforcement.py::test_subprocess_without_repo_checkout_uses_caller |
| D-18 | medium | `depth/backends/ensemble.py:852-873; lux_depth_v3/pipeline_coordinator.py:936` | DepthEnsembleBackend.ensure_available() reports ready when no constituent can run | partly fixed: f1345e772. Remaining: Deferred: (1) requiring >=2 *ready* constituents would turn today's successful single-runnable-model ensemble runs into coordinator fallbacks, so the owner must decide whether a  |
| D-16 | medium | `depth/backends/da3.py:1151; ensemble.py:490-493` | DA3 in-process engine is rebuilt on every compute(image, device=...) | fixed: 934dadd01 |
| D-08 | medium | `depth/backends/cache.py` | DepthCacheWriter: (a) :95 writes input_size: null; :182 does tuple(None); the TypeError is swallowed at :135. Metric entries written without input_size (the dataclass default) are permanentl | fixed: 3f70d27a6 |
| D-02 | medium | `depth/pipeline.py:382-439, :719-757, :871-872` | batch paths skip the depth post-processing that process_render applies | fixed: 56dd5fb2b |
| D-03 | medium | `depth/utils/depth_utils.py:263-271; depth/pipeline.py:228-232` | smooth_depth(method="median") raises for float32 depth at the default sigma; the pipeline swallows it | fixed: 8bd054504 |
| D-10 | medium | `depth/processors/depth_aware_denoise.py:155-158,187-188; config/default_config.yaml` | DepthAwareDenoise quantises float images to 8 bits and destroys any frame with a sample above 1.0 | partly fixed: c20937553. Remaining: Deferred: the 8-bit quantisation (128 -> 51 levels) and -0.5 LSB truncation bias of ordinary in-range float frames; the owner must decide whether to run cv2.bilateralFilter on fl |
| D-11 | medium | `depth/processors/base.py:82-95; depth/processors/atmospheric_effects.py:241-255; depth/processo` | processor config overrides leak permanently; short zone_params blacks out a depth zone | fixed: 1e74bf965 |
| D-25 | medium | `depth/processors/zone_tone_mapping.py:231-269; config/default_config.yaml` | default agx / reinhard zone tone curves map SDR white to 0.5 | deferred: default zone tone curves are a product decision about the look |
| D-19 | medium | `depth/depth_aware_dof.py:376-380` | depth-aware DOF divides float TIFF sources by their maximum and writes them out that way | fixed: 4eaa6720b |
| D-20 | medium | `depth/depth_processor.py:72-77; depth/pipeline.py:103-126` | DepthProcessor ignores device, silently changes the variant, loads the model twice, reports zero cache stats | partly fixed: b2ce5ee3e. Remaining: Deferred: the "vitl" comment implies the Large model was intended, but it has always loaded SMALL; the owner must decide whether DepthProcessor should switch to LARGE (changes mo |
| D-13 | medium | `depth/tools.py:783` | *_depth16.tiff depth maps are never matched | fixed: by an earlier fix in this audit. Already fixed at the start point by e45ea95: _depth_map_base strips the suffix with the anchored regex _depth16\.[^.]+$, and the repro now pairs villa_depth16.tiff wit |
| D-14 | medium | `depth/tools.py:181-182` | depth tools caches retain every full-resolution depth map and mask with a 0 % hit rate | partly fixed: cbca04d1e. Remaining: Deferred: load_depth_normalized quantises 16-bit depth to 256 levels via a uint8 resize whenever target_size is given, even at native size (what the batch always passes); the own |
| D-04 | medium | `depth/utils/depth_utils.py:446-463` | inpaint_depth_holes returns all zeros for NaN holes and rescales valid pixels for 0-holes | fixed: 978215e66 |
| D-24 | medium | `depth/utils/image_utils.py:74-77` | load_image / save_image choose the value range from the data instead of the dtype | partly fixed: 50b2b6ef9. Remaining: Deferred: truncation vs rounding in save_image (0.999 -> 254); the owner must decide whether to switch to rint, which changes the bytes of every float image the depth pipeline sa |
| D-09 | low | `depth/utils/cache.py` | DepthCache: Evidence: CONFIRMED — repro_depth_cache.py | partly fixed: 140442440. Remaining: Deferred: (1) the legacy namespace=None key still ignores shape/dtype because tests/test_da2_inference.py pins it as md5(image.tobytes()) - the owner must decide whether to drop  |
| D-05 | low | `depth/utils/depth_utils.py:192-196` | create_depth_zones(method="log") leaves zero-weight pixels and collapses into the last zone | fixed: d55a91e9f |
| D-06 | low | `depth/utils/depth_utils.py:251-261` | smooth_depth(method="bilateral") renormalises to [0,1] and quantises to 8 bits | partly fixed: 86dc7b1da. Remaining: Deferred: the 8-bit quantisation (255 levels over the span); the owner must decide whether to run cv2.bilateralFilter on float32 data (sigmaColor = edge_preserve * span), which c |
| D-07 | low | `depth/utils/depth_utils.py:497-500` | create_depth_of_field_map blows up at depth 0 | deferred: Reproduced: create_depth_of_field_map returns 1.56e9 px at depth 0 because it treats normalised [0,1] depth/focus as metres (denominator depth*(D-f/1000)+1e-8) and mixes mm and m in the numerator, so  |
| D-21 | low | `models/hf_manifest_loader.py:137-140` | _common_local_root uses the first snapshots path component | fixed: 92917bd86 |
| D-26 | low | `segmentation/material_segmenter.py:136-147; depth_intelligence/depth_estimator.py:136-139,149-1` | safe_depth_estimation (depth/models/depth_anything_v2.py:819-836) shrinks the caller's PIL image in place, derives original_size from the shrunk image, and raises TypeError for float32 array | partly fixed: 425917ebc, 20c91b379, 852e02c9f, f44831c02, 73d57a6db. Remaining: Deferred: the backend get_cache_key implementations (da3/depth_pro/da2/ensemble) still hash tobytes() without shape/mode and DA3's omi |

### Ingest, photographic delivery, VLM captioning, recipe pipeline

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| D-05 | high | `pipeline_unified.py:1056-1057, :1101-1102` | batch outputs silently overwrite each other | fixed: a8933b330 |
| D-02 | medium | `ingest/provenance.py:362-363` | GPS hemisphere lost; shutter speed and dimensions never populated | deferred: a repair was written and then reverted in review (8ea305de1): REVERTED by b76db7f (explicit undo; a plain git revert conflicts with the later D-13 tests in tests/ingest/test_provenance_capture.py). Th |
| D-03 | medium | `ingest/metadata_service.py:350-366; IMG_0001.CR2.provenance.json` | batch sidecar collision overwrites a RAW file's provenance | fixed: 4fd49280e |
| D-04 | medium | `ingest/normalize_machine_json.py:16-24, :56-60; raw_sidecar.py:24-27` | ingest_v1 normalisation keeps volatile exiftool tags | deferred: needs a new ingest normalisation profile id (hashes change) |
| D-06 | medium | `ingest/upload_staging.py:613-616; app.py:3845-3860` | upload cleanup deletes a batch a job still uses via a subdirectory | fixed: 528f77656 |
| D-01 | medium | `ingest/upload_staging.py:342; app.py:9944` | path-prefix upload conflicts escape as raw OSError | fixed: 460038b9b |
| D-07 | low | `vlm_captioning/fastvlm_runtime.py:346-381; lux_depth_v3/orchestrator.py:7183` | FastVLM decode error breaks the "structured advisory result" contract | fixed: 094e1e4b5 |
| D-15 | low | `vlm_captioning/fastvlm_runtime.py:356-359, :347-355` | timeout discards partial output; captured output is unbounded | fixed: 094e1e4b5 |
| D-16 | low | `vlm_captioning/image_proxy.py:114` | VLM proxy is blank for 16-bit grayscale and float sources | partly fixed: 775bbcca8. Remaining: Deferred: modes I (32-bit int) and F (float) still go through convert("RGB"), which treats them on a 0..255 scale; the owner must decide their full-scale convention (e.g. F in [0 |
| D-08 | low | `config_loader.py:242-245, :324; pipeline_unified.py:231-232, :326-327, :340-341, :1074-1075, :1` | recipe validation crashes, or passes recipes that cannot run | partly fixed: 66e943dfd. Remaining: Deferred: output format "exr" still validates but cannot be written by PIL; the schema in utils/recipe_validator.py (pinned by tests/utils/test_recipe_validator.py) lists exr in  |
| D-09 | low | `config_loader.py:126-139` | relative-path resolution matches keys by substring | fixed: 66e943dfd |
| D-10 | low | `ingest/validator.py:85-97; ingest/schemas.py:103, :132, :264, :324` | validation exit code follows values embedded in messages | fixed: b4f15be81 |
| D-11 | low | `ingest/validator.py:213-216, :232` | non-object JSON sidecar leaks an AttributeError | fixed: b4f15be81 |
| D-12 | low | `ingest/sidecar.py:58, :100; ingest/validator.py:213` | sidecar I/O uses the locale encoding | fixed: 36f071d54 |
| D-13 | low | `ingest/provenance.py:130-140; ingest/raw_sidecar.py:107-108; ingest/service.py:325-332` | filename starting with - reaches exiftool as an option | fixed: a788c7c7c |
| D-14 | low | `ingest/schemas.py:168-170, :173, :175-176, :180; ingest/provenance.py:349-361` | numeric-looking EXIF strings fail the whole extract | fixed: 6807c0ca8 |
| D-17 | low | `vlm/quality_validator.py:349, :426, :438-464, :520-533; vlm/scene_analyzer.py:191, :311-316, :3` | vlm/ text heuristics give wrong verdicts | partly fixed: 8480932e1. Remaining: Left as is: in non-strict detailed mode an overall mean >= 7 still passes with one aspect at 3/10; this is pinned by test_validate_strict_mode_fails_when_any_aspect_fails, so the |
| D-18 | low | `ingest/upload_staging.py:627-636` | interrupted upload batches are never reaped | fixed: 856ff56b7 |
| D-19 | low | `ingest/batch.py:160, :181` | ingest batch outputs are not atomic; a failed re-run leaves a stale manifest | fixed: 88cb55b9c |
| D-20 | low |  | evidence builder accepts exit_code: true and a missing error key | fixed: 9e277d751 |
| D-21 | low | `photographic_delivery/cohesion.py:276-282` | cohesion_diagnostics raises KeyError instead of ValueError | fixed: 983cc6dcc |
| D-22 | low | `pipeline_unified.py:704-715` | "4K upscaling" fallback shrinks images | fixed: 86614c68d |
| D-23 | low | `pipeline_unified.py:869-871` | depth stage keeps every depth map and never reads them | fixed: 86614c68d |

### spatial_ai package

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| S-1 | high | `spatial_ai/segmentation/tiling/merger.py:72-86, :329-336, :212-232; spatial_ai/segmentation/sam` | tile merger fuses whole/part masks across a seam, then destroys the fused mask | fixed: 663930fd5 |
| S-2 | high | `spatial_ai/segmentation/tiling/merger.py:332-337; spatial_ai/segmentation/tiling/validator.py:3` | seam validator aborts the run for pairs the merger itself decided to merge | partly fixed: d0acbc0f2. Remaining: Deferred, owner decision: a pair that still disagrees after the retry (IoU >= 0.35, continuity < 0.75) aborts the image as before; the options are not merging such pairs in the m |
| M-1 | high | `spatial_ai/orchestration/pipeline.py:1036-1044; spatial_ai/materials/contracts.py:86-88; spatia` | classifier labels are never valid material hints, so the materials stage fails | fixed: 543ff4fcc |
| R-1 | high | `spatial_ai/reconstruction/gaussian_rasterizer.py:326-331` | render_gaussians composites in the wrong order: far splats occlude near ones | fixed: 891c3dd65 |
| I-1 | high | `spatial_ai/ingest/linear_decoder.py:497-499, :517-520` | 16-bit RGB PNG is silently decoded at 8 bits, or rejected as "8-bit" in strict mode | fixed: 55d6526e3 |
| I-2 | medium | `spatial_ai/ingest/linear_decoder.py:49` | NaN, Inf and negative samples pass linear ingest; the validator is imported but never called | fixed: 8bcc84b1a |
| S-3 | medium | `spatial_ai/segmentation/tiling/planner.py:53-56` | planner emits redundant sliver tiles, down to 1×1 px | fixed: 6a252311a |
| S-4 | medium | `spatial_ai/segmentation/sam2_backend.py:520-530; spatial_ai/segmentation/tiling/merger.py:309-3` | every tile instance is a full-tile float32 array | fixed: 74da996af |
| S-5 | medium | `spatial_ai/segmentation/sam2_backend.py:942-953; spatial_ai/segmentation/contracts.py:157-159` | one 1-pixel-wide SAM2 mask crashes auto segmentation; bbox width/height are off by one | fixed: c9044a257 |
| O-1 | medium | `spatial_ai/orchestration/progress_tracker.py:141-152, :322-334; spatial_ai/orchestration/pipeli` | reused pipeline reports negative execution_time and progress above 100 % | fixed: 9e6072df1 |
| O-2 | medium | `spatial_ai/orchestration/error_handler.py:335-343` | GPU-OOM detection matches the substring "oom" | fixed: 51484b445 |
| G-1 | medium | `spatial_ai/orchestration/graph/artifact_store.py:693-696; spatial_ai/orchestration/graph/execut` | a corrupt committed cache artifact is never repaired | fixed: 8e5da3dbe |
| G-2 | medium | `spatial_ai/orchestration/graph/stage_adapters.py:229-235, :387-405` | cache keys omit output-affecting identity; cache hits skip sidecar writes | fixed: adf3a18d9 |
| G-3 | medium | `spatial_ai/orchestration/graph/executor.py:304-320; spatial_ai/orchestration/graph/artifact_sto` | caching is not transparent to stages | fixed: e718cb591 |
| P-1 | medium | `spatial_ai/orchestration/pipeline.py:621; spatial_ai/reconstruction/scene_builder.py:206` | multi-view loaders divide every image by 255 | fixed: a7820be6c |
| R-2 | medium | `spatial_ai/reconstruction/gaussian_rasterizer.py:417-422; spatial_ai/reconstruction/gaussian_ba` | capped renderer truncates before culling; zero-depth pixels crash or pollute the scene | fixed: d52709447 |
| R-3 | medium | `spatial_ai/reconstruction/gaussian_backend.py:826-851` | legacy RMSE and convergence do not describe the saved scene | fixed: 11bacb511 |
| R-5 | medium | `spatial_ai/reconstruction/scene_builder.py:287, :314-320` | camera path interpolates the world-to-camera translation, not the camera centre | fixed: 0d42c4188 |
| R-6 | medium | `spatial_ai/reconstruction/geometric_validator.py:225-226, :182-187, :146` | validator counts splats behind the camera and fabricates a 0.0 reprojection error | partly fixed: 2c7b6dd96. Remaining: Deferred: compute_reprojection_error still returns a fabricated 0.0 when points_2d is None, pinned by test_cpu_fill_gaps.py::test_geometric_validator_reprojection_and_coverage_ed |
| R-8 | medium | `spatial_ai/reconstruction/gaussian_rasterizer.py:347-374` | per-splat Python loop with in-place writes | fixed: 65ff9b4bd |
| M-2 | medium | `spatial_ai/materials/heuristic_fallback.py:177, :272; spatial_ai/orchestration/pipeline.py:1034` | a Python call per pixel, repeated per segment on the full image | partly fixed: 1c71ac62d. Remaining: Deferred: the pipeline still regenerates full-image heuristic intermediates once per segment; routing it through PBRGenerator._generate_batch_heuristic_shared would replace the p |
| O-3 | low |  | OOM on the last attempt rebuilds the backend on CPU and never runs it. | fixed: 359fe12e4 |
| O-4 | low |  | ErrorHandler(max_retries=0) never calls the function. | fixed: 359fe12e4 |
| O-5 | low | `spatial_ai/orchestration/progress_tracker.py:179-181; tests/spatial_ai/orchestration/test_progr` | estimated_remaining is always 0 (for s in self._stage_metrics if s not in self._stage_metrics). | fixed: 6c69692c5 |
| G-4 | low |  | _check_cache_size walks the whole cache on every store while holding the key lock. | fixed: be629f21a |
| G-5 | low |  | Graph mode fabricates result metadata. | partly fixed: 92580270b. Remaining: Deferred: graph mode names the segmentation stage "segment" in stages_completed/stage_reports while imperative mode records "segmentation"; the graph name is pinned by tests/spat |
| P-2 | low |  | Empty YAML sections crash. | fixed: 9e0f208d2 |
| I-3 | low |  | Palette PNG, MinIsWhite TIFF and multi-page TIFF decode to wrong pixels without error. | fixed: 6463c154d |
| I-4 | low |  | EXIF extraction walks only IFD0, so exposure, lens, date and GPS are always missing. | fixed: 951e4fe41 |
| I-5 | low |  | Manifest validation is fail-open. | partly fixed: e7ebf9729. Remaining: Deferred: unknown fields are still silently dropped by pydantic (extra="ignore") and lost on a write round trip; the owner must choose between preserving them (extra="allow", add |
| I-6 | low | `spatial_ai/ingest/validators.py:119; linear_decoder.py:147; manifest_schema.py:119` |  | fixed: 2f49a110d |
| S-6 | low |  | ArrayDigestCache keys on id(array). | partly fixed: 3074090fd. Remaining: Deferred, owner decision: an array mutated in place keeps its cached digest, since detecting that needs a rehash that removes the N-3 memoization pinned by test_same_array_hits_c |
| S-8 | low |  | track_temporal hands a departed object's ID to a new object. | fixed: 0ab1229cc |
| R-4 | low |  | The only RNG use (init subsample) runs before seeding. | fixed: 65ed325e0 |
| R-7 | low |  | Exports write in place. | fixed: 0b865be4a |
| M-3 | low |  | Linear albedo is quantised to 8 bits. | fixed: 04b66dd4f |

### Image and video pipelines, processors, streaming, utils

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| C-01 | critical | `tp/processors/luxury_video_master_grader.py:628-647` | grader filtergraph is invalid for every preset | fixed: 0901403fb |
| H-01 | high | `tp/pipelines/rendering_4k/stages.py:35-113; rendering_4k/pipeline.py:482-483, 518-519` | HDR tone curves applied to display-referred 8-bit input | deferred: tone curves on display-referred input is a decision about the look |
| H-02 | high | `tp/pipelines/rendering_4k/stages.py:286-287` | .cube LUT applied with red and blue swapped (4K pipeline) | fixed: 5b3815b64 |
| H-03 | high | `tp/streaming/stages.py:879, 895-904` | same LUT swap plus posterisation in the streaming stage | fixed: 1a90e7872 |
| H-04 | high | `tp/pipelines/pro_pipeline.py:642, 705, 771` | Gaussian blur runs across the colour channels | fixed: af621c21e |
| H-05 | high | `tp/utils/image_utils.py:25; pipelines/unified_luxury_pipeline.py:789-791; pipelines/pro_pipelin` | 16-bit greyscale inputs load as almost pure white | partly fixed: 22adf7ea0. Remaining: Deferred: 32-bit integer TIFFs (mode I) and float TIFFs (mode F) still clip via Pillow because their full scale is not defined by the file; the owner must decide the convention ( |
| H-06 | high | `tp/processors/luxury_video_master_grader.py:30` | grader preset LUT paths point at a directory that does not exist | fixed: bdc9b4930 |
| H-07 | high | `tp/processors/luxury_video_master_grader.py:543-612, 664-671` | the 10-bit "master" chain is squeezed through 8-bit | deferred: replacing eq/gradfun redefines the grade maths (the look) |
| H-08 | high | `luxury_tiff_batch_processor/pipeline.py:288-320; profiles.py:76-82` | TIFF batch processor writes 16-bit RGB masters as 8-bit | fixed: 85f2066e0 |
| M-01 | medium | `tp/pipelines/rendering_4k/pipeline.py:824-836` | depth cache returns another image's depth map | fixed: 360a7f24f |
| M-02 | medium | `tp/pipelines/unified_luxury_pipeline.py:1427; rendering_4k/pipeline.py:997-1049; pro_pipeline.p` | same-stem inputs silently overwrite each other's deliverables | fixed: 85e89b2d2 |
| M-03 | medium | `tp/pipelines/parallel_io.py:172-226` | Ctrl+C during a parallel-I/O batch deadlocks | fixed: 1ad87fc13 |
| M-04 | medium | `tp/pipelines/depth_tools.py:729, 355-390, 424-438; depth/tools.py` | legacy depth_tools mis-pairs and drops files | fixed: 8c2d75355 |
| M-05 | medium | `tp/pipelines/depth_tools.py:320-349, 658-675` | DOF mode is about 10× slower than it needs to be | deferred: DOF filter choice changes the look |
| M-06 | medium | `tp/pipelines/quality_feedback_bridge.py:101-119, 343-388; tp/pipelines/rendering_4k/quality.py:` | the LPIPS/perceptual path can never run | partly fixed: 538e3e136. Remaining: Deferred: emitting null instead of 0.0, since QualityMetrics (rendering_4k/types.py:205-208) types them as float, test_rendering_4k_quality.py:72 pins 0.0 for the unassessed case |
| M-07 | medium | `tp/pipelines/rendering_4k/pipeline.py:986-1019; tp/pipelines/pro_pipeline.py:247, 787-853` | preserve_metadata=True preserves nothing | fixed: 8ea6d40d1 |
| M-08 | medium | `tp/pipelines/rendering_4k/pipeline.py:1259-1278; tp/pipelines/pro_pipeline.py:1083-1089; lux_re` | batch CLIs exit 0 when every image failed | fixed: 19bdc8a0b |
| M-09 | medium | `tp/pipelines/unified_luxury_pipeline.py:115` | EXR is advertised but cannot be loaded | fixed: c470a4d27 |
| M-10 | medium | `luxury_tiff_batch_processor/io_utils.py:506-543, 594-606` | TIFF batch processor loses DPI and duplicates the ICC tag on the 16-bit path | fixed: cb636b14e |
| M-11 | medium | `tp/processors/luxury_video_master_grader.py:873` | --tone-map cannot be switched off correctly | fixed: 84d5d25fb |
| M-12 | medium | `tp/processors/luxury_video_master_grader.py:889-906, 1021-1026, 94, 133, 160-164` | preset halation and deband never apply; --deband soft crashes ffmpeg | fixed: 67e28fe73 |
| M-13 | medium | `tp/streaming/stages.py:715-718` | MaterialResponseStage breaks on greyscale images | fixed: 8d25abf9f |
| M-14 | medium | `tp/streaming/stages.py:992-1029` | ResizeStage quantises float/16-bit data to 8-bit | fixed: d93e088ac |
| M-15 | medium | `tp/streaming/stages.py:1114-1139` | depth-adaptive denoise skips the far field; more strength means less denoising | fixed: cc1a632a7 |
| M-16 | medium | `tp/streaming/checkpoint.py:83-84, 209-215` | checkpoints are written in place and one bad file breaks resume | fixed: dcd5968f6 |
| M-17 | medium | `tp/utils/format_utils_enhancements.py:369, 305, 444, 113-114` | truncated literals in format utilities | fixed: 1b25f607e |
| M-18 | medium | `tp/utils/format_utils_enhancements.py:213-232, 443-448, 585-589` | smart_convert(preserve_bit_depth=True) writes 8-bit for 16-bit RGB TIFF | fixed: 1b25f607e |
| M-19 | medium | `tp/utils/security.py:231, 325-338, 353-381` | FFmpeg filter "validation" does not contain file access | partly fixed: 7fd844ae8. Remaining: Deferred: the owner must decide whether to replace this denylist of documented path options with a filter-name allowlist, and whether other additional_args such as extra position |
| M-20 | medium | `tp/pipelines/lux_render_pipeline.py:945-957, 1036-1047` | PLAUSIBLE: SDXL branch of lux_render calls the wrong pipeline | not confirmed: plausible only (cannot run without weights) |
| L-01 | low | `tp/pipelines/lux_render_pipeline.py:692-705` | adjust_contrast_saturation pivots "contrast" on the per-pixel channel mean, so it is a second saturation multiplier and does nothing to neutral tones | deferred: contrast pivot is a decision about the look |
| L-02 | low | `lux_render_pipeline.py:729, 740-741` | ImageFont.truetype("arial.tt") typo forces the tiny default font; RGBA fill on RGB gives an opaque box, not 60% alpha | fixed: d17692aa3 |
| L-03 | low | `pro_pipeline.py:646, 676, 709, 746, 775, 821, 825; unified_luxury_pipeline.py:1315, 1388, 1547;` | quantisation by truncation, not rounding; darkening accumulates across stages | partly fixed: 5d30146e8. Remaining: The pro, unified luxury and 4K pipelines truncated float->uint8/uint16 conversions, darkening output ~0.5 level per stage (5-stage pro run on grey noise: mean 127.57 -> 125.56, n |
| L-04 | low | `pro_pipeline.py:950, 1028, 1021, 1033` | --format short flag is a bare - (meant -f); --pattern and --workers are ignored | partly fixed: 5e300db4f. Remaining: Deferred: --workers is still unused because batch_process is sequential; its help text and docs now say so, and the owner must decide whether to implement parallel batch processi |
| L-05 | low | `depth_tools.py:949-954, 778, 1003` | documented dof and --falloff are rejected; only do / --fallof work | fixed: 96157f553 |
| L-06 | low | `unified_luxury_pipeline.py:1589, 1626, 1662, 1697; rendering_4k/stages.py:414-415` | int(size*ratio) makes the long edge target−1 for many sizes | fixed: ecb2b5795 |
| L-07 | low | `luxury_video_master_grader.py:417-441` | --target-fps 23.976 (the help text's example) becomes 2997/125, not 24000/1001; inf raises uncaught OverflowError | fixed: a444719da |
| L-08 | low | `streaming/async_pipeline.py:815-829, 916-930` | stop_on_error=True never stops the batch | fixed: a765fbffd |
| L-09 | low | `streaming/streaming.py:169-173, 220` | get_stats()['items_processed'] saturates at window_size | fixed: fb32f87b6 |
| L-10 | low | `streaming/async_pipeline.py:661, 776-778` | _current_memory_mb is never incremented, so max_memory_mb enforces nothing | fixed: 548a9ea44 |
| L-11 | low | `utils/security.py:428-436` | nested timeout() cancels the outer alarm | fixed: 4866abe48 |
| L-12 | low | `utils/recipe_validator.py:283-287, 340-351` | a missing custom schema path silently falls back to the built-in schema; the no-jsonschema fallback has a different stage list | partly fixed: 04392254e. Remaining: Deferred: a missing custom schema_path still silently uses the built-in schema because tests/utils/test_recipe_validator.py::test_nonexistent_schema_path_uses_default deliberatel |
| L-13 | low | `utils/input_validation.py:438, 143` | empty float array raises ValueError instead of returning a result; SUPPORTED_MODES lacks I;16, so the repo's own *_depth16.png fails default validation | fixed: 5dbc408a8 |
| L-14 | low | `processors/material_response/core.py:973-989, 31-35` | apply_transformation_tensor casts the matrix to the image dtype (uint8 × 0.5 gives black) and applies pixel @ M, the transpose of the row-equals-output convention; the validator rejects nump | fixed: 1534eccb6 |
| L-15 | low | `processors/material_response/core.py:92-108` | pure-Python O(N⁴) DFT in a module that already imports numpy | fixed: 1534eccb6 |
| L-16 | low | `enhancers/enhance_pool_aerial.py:6, 18-46` | a script inside the package: bare sibling import, plus mkdir and image processing at import time | partly fixed: d90a225a4. Remaining: Deferred: main() still cannot run because _assign_full_image, _cluster_stats, _downsample_image and assign_materials no longer exist in board_material_aerial_enhancer (and enhanc |
| L-17 | low | `lux_render_pipeline.py:996-1006, 1072-1087` | PLAUSIBLE (not run): --w4k is a no-op. target_min is used only inside the Real-ESRGAN branch, which is always off, and the "Using Pillow Lanczos fallback" the warning promises is not impleme | not confirmed: plausible only |
| L-18 | low | `rendering_4k/quality.py:380-408` | PLAUSIBLE (arithmetic only): the noise weight is added to the denominator but only ever subtracts, so a perfect image scores at most 0.85 against a 0.75 threshold | not confirmed: plausible only |
| L-19 | low | `luxury_video_master_grader.py:630-636, 1173-1177` | PLAUSIBLE (not run): shlex.quote is not filtergraph escaping, so LUT paths containing ,, : or ' should break the graph; ffmpeg has no timeout and writes straight to the final path, so a fail | not confirmed: plausible only |

### dashboard/ and rl/ packages

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| D1 | high | `dashboard/execution_manager.py:405, 418-453; execution_graph/scheduler.py:340-350` | pipeline nodes run 2^(k-1) times; cyclic or dangling pipelines report "complete" with nothing run | fixed: 1b8fcd6e2 |
| R1 | high | `rl/ma_qmix.py:200; rl/ma_trainer.py:217; rl/ray_distributed.py:266` | CentralCritic.forward() raises on its first call; QMIX training never happens | fixed: 63dd76db6 |
| R2 | high | `rl/ma_optimize.py:165, 182, 189-192; rl/ma_trainer.py:90-107` | optimize_multi_agent() crashes on its second training update, always with the default iterations=20 | fixed: 772a9563a |
| R4 | high | `rl/pipeline_foundation.py:440-443` | PipelineFoundationTrainer warm-up drives the learning rate to exactly 0 and never restores it | fixed: 5e07ba90f |
| R7 | high | `rl/gat.py:169-188; rl/ma_gat_critic.py:126-141; rl/temporal_gat_critic.py:139-153` | GAT critics ignore almost all node features | fixed: 985b375aa |
| R13 | high | `rl/mcts.py:57-71, 243, 249-279, 417-425, 429; rl/mcts_node.py:160-172` | mcts.MCTS does not search; MCTSPipelineOptimizer returns the input pipeline unchanged | fixed: 1ac4c9fc1 |
| D2 | medium | `dashboard/execution_api.py:43-58` | one non-JSON-serialisable node output silently unsubscribes every WebSocket client; GET run returns 500 | fixed: 7620fbea2 |
| D10 | medium | `dashboard/rl_api.py:72-119; rl/optimize_rl.py:56-57, 148-152` | POST /rl/optimize runs the whole training on the event loop and writes into the server's working directory | fixed: f3bca95c0 |
| D11 | medium | `dashboard/execution_manager.py:420` | a cancel issued while node k runs takes effect only after node k+1 has run | fixed: 3bc774201 |
| D3 | medium | `dashboard/experiment_api.py:228-249` | a NaN or Infinity metric permanently breaks every read endpoint of that experiment | fixed: 6d6165a3d |
| D4 | medium | `dashboard/experiment_api.py:235-249` | log_metrics() read-modify-write is not atomic; concurrent writers lose metrics | fixed: 6d6165a3d |
| D5 | medium | `dashboard/server.py:223-233; dashboard/server.py:170-173; dashboard/node_api.py:156` | POST /api/event bypasses the history cap; limit=0 returns everything; client-supplied strings are rendered with innerHTML | fixed: 4ac7d0074 |
| D6 | medium | `dashboard/server.py:124-133` | broadcast_event() from any non-main thread never reaches live clients, silently | fixed: 4ac7d0074 |
| D7 | medium | `dashboard/artifact_api.py:173-218; dashboard/node_api.py:217-275; dashboard/artifact_preview.py` | artifact preview endpoints: unbounded reads, valid UTF-8 reported as binary, thumbnails return 500 | fixed: a4bd2b42b |
| D8 | medium | `dashboard/dag_editor_api.py:115-122` | DAG editor: a saved pipeline cannot be loaded back | fixed: d6d514191 |
| D9 | medium | `dashboard/studio_inspector.py:62-67, 1142-1145; dashboard/artifact_preview.py:290-294; dashboar` | every 3D viewer in the dashboard is dead: the three.js add-on URLs do not exist for the pinned version | fixed: 5742f5e00 |
| R5 | medium | `rl/pipeline_foundation.py:139-144, 405-411, 414` | GraphEncoder adjacency mask: the documented shape crashes, the accepted shape produces NaN; InfoNCE keeps the diagonal | fixed: fd79c554f |
| R6 | medium | `rl/gat.py:177-178; rl/ma_gat_critic.py:74, 100-105` | GATLayer crashes whenever dropout > 0 | fixed: 2bd079316 |
| R8 | medium | `rl/gat.py:171-174, 184-185; rl/ma_gat_critic.py:169-176; rl/temporal_gat_critic.py:180-190` | GAT critics: Python loops per batch item × timestep × layer × (nodes + edges) | fixed: b0306cbb1 |
| R9 | medium | `rl/ma_gat_critic.py:95, 112-124; rl/temporal_gat_critic.py:101, 125-137; rl/ma_qmix_attn.py:128` | attention and GAT critics cannot be fed with the package's own state encoders | fixed: 247c5c6bd |
| R10 | medium | `rl/world_model.py:124-133, 187-212; rl/model.py:256-263, 278-283` | WorldModel.predict() is non-deterministic; save()/load() drop architecture fields | fixed: 73e631fec |
| R11 | medium | `rl/gt_muzero.py:312-336` | GraphTransformerMuZero.recurrent_inference ignores the action and the hidden state | deferred: conditioning the dynamics on the action is an architecture change |
| R14 | medium | `rl/muzero_mcts.py:56-67` | muzero_mcts: selection ignores the predicted reward; temperature handling crashes; search runs with autograd on | fixed: a777b3929 |
| R15 | medium | `rl/cost_reward.py:196-199` | CostAwareRewardShaper penalties accumulate over the shaper's lifetime | fixed: 3ca79f545 |
| D12 | low | `dashboard/optimization_api.py:265; rl_api.py:45` | POST /optimize/stop/{id} has no effect | fixed: f43e68cb4 |
| D13 | low | `dashboard/dag_editor_api.py:34, 107-112, 204; dashboard/experiment_api.py:32, 140, 175; rl/opti` | dashboard defaults write relative to the process CWD; the dashboard tests already leave a stray pipelines/ in the repo root | fixed: 837251bd7 |
| R3 | low | `rl/ma_optimize.py:227, 337; rl/optimize_rl.py:159` | zero iterations crash instead of returning an empty result | fixed: f03827f90 |
| R12 | low | `rl/muzero_transformer.py:143, 157-162` | MuZeroTransformer.forward(mask=...) cannot take the mask it documents | fixed: 89c14a848 |
| R16 | low | `rl/muzero_trainer.py:180; rl/ma_comm.py:137-144, 158-162; rl/ma_optimize.py:161-162, 205-206` | smaller defects, each confirmed unless marked | partly fixed: b08ad1af1. Remaining: Fixed: RLTrainer replay entries now store detached log_prob/value; PrioritizedReplayBuffer uses (\|delta\|+1e-6)**alpha (signed/all-zero TD errors no longer crash); MuZero policy |

### Evaluation, metrics, perceptual, foundation, dev tooling, auxiliary packages

| ID | Severity | Where | Finding | Status |
| --- | --- | --- | --- | --- |
| M-01 | high | `evals/apex_harness.py:206; evals/metrics.py:400-401; evals/vision_language/llava_schema.py:143-` | NaN and off-scale scores are clamped to a perfect 1.0 | partly fixed: f6e0fdb2f. Remaining: Deferred: finite off-scale scores (3, 35) are still clamped to 1.0 because test_llava_schema.py::TestParseVqaResult::test_score_clamping, tests/validation/test_llava_quality_vali |
| M-02 | high | `evals/metrics.py:71-75` | _to_numpy decides the data range from pixel content | partly fixed: 495647ad1. Remaining: Deferred (needs a new threshold or an explicit value-range argument, per the pinned test_high_value_normalization convention): two float images that BOTH overshoot 1.0 are still  |
| M-03 | high | `evals/metrics.py:303-340, :343-378` | IoU and Dice return 1.0 for disjoint label masks | fixed: d9efda9f6 |
| M-22 | high | `dev/refactor_engine.py:382-383, :333-349, :388-389, :415-416, :468-484, :199; scripts/dev/run_r` | AutoRefactorEngine produces un-importable packages and reports success | fixed: 8d2f37304 |
| M-04 | medium | `perceptual/metrics.py:475-484, :495-501, :504-510, :513-519; perceptual/__init__.py` | four exported perceptual convenience functions raise TypeError on every call | fixed: 7f7a68aec |
| M-05 | medium | `evals/metrics.py:263-268; evals/benchmark_suite.py:212-222, :295-298` | unavailable LPIPS earns full LPIPS credit | partly fixed: 91cb49745. Remaining: Deferred: lpips_score itself still returns 0.0 when lpips/torch are missing because tests/evals/test_metrics.py::TestLPIPS::test_lpips_unavailable pins a float return ("returns 0 |
| M-06 | medium | `evals/benchmark_suite.py:341-348` | run_benchmark_batch means are poisoned or diluted | fixed: 2dd8df206 |
| M-07 | medium | `evals/apex_harness.py:290-294, :310-322` | fail_on_vlm_error=True does not fail; unreadable deliverables are skipped | fixed: f605da760 |
| M-08 | medium | `evals/vision_language/llava_scoring.py:77-78; llava_schema.py:41` | VQA gate ignores unknown severities | fixed: 47504ea76 |
| M-09 | medium | `evals/auto_optimizer.py:233-239` | AutoOptimizer feeds the parent's metrics to later iterations | fixed: 58787f486 |
| M-10 | medium | `evals/auto_optimizer.py:60-88; evals/auto_opt_types.py:171-178` | composite score mixes PSNR in dB with 0..1 metrics | fixed: a23df0e05 |
| M-11 | medium | `evals/self_heal_runner.py:184-214, :190, :229, :308-309` | self_heal validates fixes that were never applied | fixed: 0fe74aca0 |
| M-12 | medium | `evals/semantic_diff.py:162-193, :208-210, :229, :285-291, :348-361` | semantic_diff: valid JSON of the wrong shape crashes with raw AttributeError or ValueError instead of SemanticDiffError. A prose-prefixed JSON answer is treated as a parse failure and return | fixed: 576a420fc |
| M-13 | medium | `perceptual/metrics.py:324-338; perceptual/analyzer.py:204-215` | BRISQUE fallback is higher-is-better but labelled lower-is-better | fixed: 786e79039 |
| M-14 | medium | `perceptual/metrics.py:142-150` | LPIPS falls back to an MSE object under the LPIPS key | fixed: 786e79039 |
| M-15 | medium | `perceptual/tracker.py:142-150; perceptual/baseline.py:215-248; baseline.py:114` | EnhancementTracker never measures improvement for the documented usage | fixed: f8438c5a3 |
| M-16 | medium | `foundation/hardware_abstraction.py:94-109, :132-176, :217-233, :244-293, :303-305` | CoreML backend is always "available" | partly fixed: a96f706b8. Remaining: Deferred the PLAUSIBLE item: nn.Module arguments are still moved in place by .to(device), so after a fallback the caller's model stays on the fallback device; the owner must deci |
| M-17 | medium | `foundation/device_manager.py:335-361` | M4-Max core counts are reported on every non-Darwin host | fixed: b95294ebf |
| M-18 | medium | `foundation/memory_manager.py:262-288, :117-146, :466-495, :412-430; foundation/substrate.py:243` | MemoryManager: A double deallocate puts the same tensor in the pool twice, so two later allocations are the same object. | partly fixed: b858a1ec1. Remaining: Deferred: on CPU with profiling off the usage estimate is still 0 (nothing is tracked), so the watermarks cannot fire there; the owner must decide whether byte accounting should  |
| M-19 | medium | `foundation/tensor_processor.py:139-145, :268-283` | TensorProcessor.allocate(requires_grad=True) returns a non-leaf tensor; normalize() on uint8 returns inf | fixed: bb4f969ce |
| M-23 | medium | `dev/ast_canonicalizer.py:73-78, :158-179, :215-226; dev/formatting.py:339-346; dev/refactor_eng` | Canonicalizer: The filter if k is not None makes the "all constant keys" guard vacuous, so a dict containing a spread is reordered. | fixed: 1610e7faf |
| M-24 | medium | `dev/test_markers.py:60, :415-416, :306-310, :77-83, :143-147; /ci/stress/repo/tests/test_a.py` | marker retrofit writes un-collectable test modules | fixed: 8bfd99208 |
| M-25 | medium | `dev/check_test_markers.py:284-286, :102-106, :421-429, :221-229, :157-176` | marker checker: A missing tests root (for example, the wrong working directory with the relative default tests) yields an empty report, 100% coverage and exit 0, even with --strict. | fixed: a3ae0b8fc, 4c6b05e63, aeff208ae |
| M-26 | medium | `dev/ast_index.py:156-186; dev/ast_hash.py:200-232; app.py` | ASTEquivalenceIndex re-parses the whole file once per function and class | fixed: 9365aced8 |
| M-28 | medium | `neuroaesthetics/spatial_frequency.py:145-151, :184-186, :163-172` | SpatialFrequencyAnalyzer counts the DC term as low-frequency energy and measures radius in FFT bins | fixed: 69e4329be |
| M-29 | medium | `comfyui/custom_nodes.py:76-81, :246, :287; atmosphere/sky_blending.py:150, :266, :325-335, :64-` | SkyBlender and the ComfyUI SkyGANNode | partly fixed: 5c6c48545. Remaining: Deferred: (1) the mu-law(5000)+gamma tone map lifts every non-sky pixel (sRGB 25->176); the owner must choose an invertible tone curve for LDR sources (e.g. inverse of the 2.2 li |
| M-30 | medium | `evals/metrics.py:219-231; evals/apex_visual.py:668-688` | the evals.metrics.ssim fallback is not SSIM | fixed: 4a89c7659 |
| M-20 | low | `perceptual/image_loader.py:117-123` | perceptual ImageLoader: metadata is read after convert("RGB"), so bit depth, colour space and format are always 8, RGB and "unknown". 16-bit greyscale input saturates. normalize=False crashe | fixed: 234789913 |
| M-21 | low | `foundation/performance_monitor.py:503-520, :135-137` | PerformanceMonitor: enabling the monitor inside a block that was entered while disabled leaves start_time at 0.0, so the recorded elapsed time is seconds since 1970. Once the 10 000 cap is r | fixed: d283524c0 |
| M-27 | low | `dev/formatting.py:162-182, :355-375, :178, :371, :210-217` | write_formatted and format_directory | fixed: 55b529779 |
| M-31 | low | `metrics/ledger.py:386-387, :458, :529; metrics/aggregator.py:76-79` | performance ledger descriptive statistics | fixed: b2b90926e |

## Preserved contracts and remaining limits

- No ingest, machine-mode, evidence, run-card or preset contract version was changed. Repairs that would have
  needed one are listed under Decisions needed.
- Lux Depth V3 remains the production default; its plan, evidence and manifest bytes are unchanged for runs that
  succeeded before the audit.
- The review read most, not all, of the repository. Each subsystem review recorded what it did not read; the
  largest unread areas are `analyzers/`, `style_transfer/`, `diffusion/`, `pfm/`, `realize_v8/`, `dwm/`, parts of
  `atmosphere/` and `comfyui/`, `reconstruction/nvdiffrec_backend.py`, and the embedded JavaScript of the dashboard pages.
