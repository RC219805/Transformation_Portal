# Spatial AI Foundation Roadmap (Architectural Constitution)


**Path:** `docs/spatial_ai/ROADMAP.md`  
**Status:** Draft (normative)  
**Owner:** Spatial AI / Data Foundation maintainers  
**Last Updated:** 2026-02-10

## Purpose

This document defines the **strategic and architectural pivot** for `Transformation_Portal`:  
from a high-fidelity rendering + depth-processing toolkit into a **Spatial AI Foundation** capable of training **World Models** that learn physics, geometry, and causality in the built environment.

This is not a backlog. This is a **constitution**: it establishes the non-negotiable invariants, the system boundaries, and the milestone sequence that keep the system “clean” and prevent **model collapse**.

Any implementation work (PRs, experiments, services) MUST align with this document. Any material deviation requires an Architecture Decision Record (ADR) and explicit sign-off.

---

## Current State Assessment

`Transformation_Portal` currently operates as a high-fidelity rendering and depth-processing engine. It already has valuable “primitive organs” (depth-aware enhancement, document provenance), but it lacks the infrastructure required for:

- petabyte-scale ingestion and training throughput
- longitudinal / multi-visit temporal indexing
- geometry-first scene representation (poses, reconstructions)
- collapse-resistant foundation training (predictive, non-recursive learning)
- physics/consistency validation loops
- secure enterprise query surfaces that never expose raw moat data

The chasm to cross: **single-image processing → longitudinal, spatiotemporal pattern recognition at scale**.

---

## North Star

Build a **Clean Spatial AI stack** from an **Exclusive High-Fidelity Spatial AI Dataset** such that:

1. The dataset remains **provenance-clean** and **high-fidelity** forever (the “Clean Moat”).
2. The training stack learns **physical grounding** (linear light + real noise + real geometry).
3. The learned representations support **spatial intelligence**:
   - object permanence
   - arrow-of-time reasoning (causal constraints)
   - robust depth under specular/transparency
   - semantically grounded retrieval via docs
4. The system resists **model collapse** by preventing synthetic pollution and by monitoring representation health.

---

## Non-Negotiable Invariants (The Constitution)

### I. Data Fidelity is Sacred
- Training inputs MUST preserve **12–14 bit** (or higher) sensor information when available.
- Training inputs MUST preserve **linear-light** relationships: pixel intensity MUST remain a linear proxy for captured light (photon count proxy), not tone-mapped or gamma-corrected.
- Any pipeline that silently converts to 8-bit, sRGB, JPEG, or tone-mapped outputs MUST be treated as **rendering-only**, not training ingest.

### II. Provenance is a Gate, Not a Note
- Any datum entering the **foundational training set** MUST have a **verified human lineage**.
- Derived artifacts (depth maps, segmentation, splats, NeRF renders, synthetic augmentations) MUST be tracked as **derived** and MUST NOT be re-ingested as “truth”.

### III. Longitudinal Structure is First-Class
- The dataset MUST be representable as:  
  **Property ID → Visit ID → Timeline Index**
- The system MUST preserve and exploit temporal constraints: *Visit A causally precedes Visit B*.

### IV. Anti-Collapse by Design
- The training curriculum MUST avoid recursive self-training on model outputs as ground truth.
- Collapse monitoring MUST exist from the first foundation run (representation variance, tail retention, anomaly detection).

### V. Reproducibility and Auditability are Required
- Every training sample MUST be traceable through:
  - content hash
  - provenance chain
  - decode pipeline version
  - transformations applied
- The system MUST be able to answer: “Exactly what data and decoding produced this model weight?”

### VI. Security and Compliance are Not Optional
- Raw moat data MUST remain access-controlled.
- Production services MUST expose **capabilities**, not raw assets.
- PII redaction MUST be enforceable (faces, license plates, sensitive documents) before enterprise exposure.

---

## System Boundaries

### In Scope
- Clean data foundation (contracts, schema, ledger/catalog, decode, throughput)
- Geometry + reconstruction pipelines (poses, depth priors, splats/NeRF assets)
- Predictive foundation models (JEPA variants: image + temporal)
- Vision-language grounding (paired documentation + spatial assets)
- “No BS” validation (consistency, physics checks)
- Secure query APIs for enterprise use

### Explicitly Out of Scope (Until the Data Foundation is Proven)
- Large-scale training runs before Milestone 2 is complete
- Any “world model” claims based on 8-bit / tone-mapped / JPEG inputs
- Any ingestion that bypasses provenance gates “for convenience”
- Any external data mixing that compromises the clean moat

---

## Architecture: What We Are Building

### Conceptual Layers
1. **Data Foundation (Clean Moat)**  
   Contracts, provenance gates, catalog/ledger, linear decode, high-throughput I/O.

2. **Spatial Intelligence Core**  
   Geometry bootstrap, 3D recon assets, predictive representation learning, causal/temporal modeling.

3. **No BS Validation + Production Layer**  
   Consistency/physics harness, multimodal grounding, secure enterprise query surface.

---

# Phase I — The Data Foundation (The “Clean” Moat)

This phase builds the irreversible moat: the system that preserves fidelity, provenance, and temporal structure.

## Milestone 0 — The High-Fidelity Contract (Schema + Provenance)

**Intent:** Define “ground truth” standards *before* ingesting scale.

**Strategic Justification:** The value is technical fidelity (RAW/TIFF) + provenance-clean status. Without a strict schema, you lose the tails of the distribution (shadow nuance, rare geometry) and invite early model collapse.

**Deliverables**
- `docs/spatial_ai/DATA_CONTRACT.md`  
  Defines canonical linear space, acceptable encodings, required metadata, forbidden transforms.
- `docs/spatial_ai/SCENE_SCHEMA.md`  
  Defines multi-visit longitudinal structure: Property → Visit → Timeline.
- Codebase: **Provenance gates** in ingestion path (hard fail if lineage is missing or unverified).

**Acceptance Gates**
- Any training-ingest sample missing required provenance fields is rejected.
- Any sample that cannot prove preserved bit-depth / linearity is rejected.
- Any derived artifact MUST be labeled as derived and cannot be promoted to “truth”.

---

## Milestone 1 — The Catalog & Provenance Ledger (Spatiotemporal Index)

**Intent:** Convert a filesystem into a queryable spatiotemporal index that encodes causal relationships.

**Strategic Justification:** The “holy grail” is causality across time. File lists do not encode “construction precedes completion”; the catalog must.

**Deliverables**
- Parquet/Iceberg-style catalog (initially Parquet is acceptable; Iceberg is the scale target):
  - indexes: `content_hash`, `capture_timestamp`, `sensor_bit_depth`
  - joins: `property_id`, `visit_id`, `timeline_index`
- Construction sequence linking:
  - explicit relationships: *framework → finished wall*, *rough-in → finish*
- Quality gate: **bit-depth enforcement**
  - reject silent 8-bit downconversions

**Acceptance Gates**
- Catalog queries can fetch:
  - all assets in a visit
  - temporal sequences across visits
  - assets matching sensor/bit-depth constraints
- Catalog refuses to index assets failing Milestone 0 contract validation.

---

## Milestone 2 — High-Bit-Depth Linear Decode Pipeline (Immediate Thin Slice)

**Intent:** Build a training ingest path that bypasses destructive 8-bit normalization.

**Strategic Justification:** JPEG/8-bit training introduces banding/blocking/clipping artifacts that models misinterpret as physics. Physical grounding requires **linear-light tensors** on GPU.

**Deliverables**
- `src/transformation_portal/io/hifi_raw.py`  
  GPU-friendly decode pipeline outputting `float16` or `bfloat16` **linear tensors**.
  - support pathways:
    - nvJPEG2000 (where applicable)
    - RAW decoders (LibRaw/rawpy or proprietary SDKs) with *linear output*
- Noise modeling:
  - sensor noise profiles so the model learns “texture vs ISO noise”
- Forked ingestion:
  - dedicated **Training Ingest** path that never passes through rendering-only sRGB normalization

**Acceptance Gates**
- A single RAW/TIFF sample can be decoded into a canonical linear tensor with:
  - preserved dynamic range
  - no tone mapping, no gamma, no 8-bit quantization
- Automated tests detect:
  - “silent 8-bit conversion” (hard fail)
  - non-linearity (hard fail)
- All decoded outputs are hash-traceable to input + decoder version.

**This is the immediate priority.**
No foundation training begins before this milestone is complete.

---

## Milestone 3 — I/O Architecture for Saturation

**Intent:** Ensure data loading does not bottleneck training.

**Strategic Justification:** Scaling only works if we saturate the accelerators. World-model training requires streaming temporal sequences, not single frames.

**Deliverables**
- `tp-bench io` benchmarking tool:
  - measures end-to-end dataloader throughput
  - reports GPU utilization proxy (data stall time)
  - supports temporal sequence streaming
- Target:
  - sustain **>90% GPU utilization** while streaming temporal sequences

**Acceptance Gates**
- Bench produces stable metrics across runs.
- Regression thresholds are defined and enforceable (performance “quality firewall” for I/O).

---

# Phase II — The Spatial Intelligence Core

## Milestone 4 — Geometry Bootstrap (Poses + Depth Priors)

**Intent:** Provide geometric priors (poses/intrinsics/depth) so models can learn 3D.

**Strategic Justification:** The archive contains non-Manhattan, organic forms. Standard planar assumptions fail. Accurate poses are required to learn continuous topology.

**Deliverables**
- SfM/SLAM integration:
  - camera intrinsics + extrinsics per visit
- Hard negative mining:
  - glass walls, mirrors, specular surfaces where depth fails
  - use these as robustness training targets

**Acceptance Gates**
- Pose outputs are versioned, cataloged, and lineage-traceable to source images.
- A “hard negatives” subset is generated reproducibly and queryable.

---

## Milestone 5 — 3D Reconstruction (NeRF + Gaussian Splats)

**Intent:** Turn multi-view imagery into renderable, queryable 3D scene assets.

**Strategic Justification:** World models must predict state, not just pixels. NeRF/splats enable continuous representations including view-dependent effects.

**Deliverables**
- `src/transformation_portal/world_assets/`
  - pipeline to bake scenes into NeRF and/or Gaussian Splat assets
- Reflection/transparency modeling emphasis:
  - glass/specular effects handled explicitly (anti-“shiny object” failure)

**Acceptance Gates**
- Novel-view render stability meets consistency harness thresholds (see Milestone 9).

---

## Milestone 6 — Image JEPA Foundation (Anti-Collapse)

**Intent:** Train a collapse-resistant image foundation via predictive representation learning.

**Strategic Justification:** Avoid recursive training on generative outputs. JEPA learns by predicting representations of missing information, grounding in physical structure.

**Deliverables**
- Spatial inpainting tasks:
  - mask load-bearing structures and predict their representation based on context
- Collapse monitoring:
  - real-time embedding variance + tail retention metrics

**Acceptance Gates**
- Monitoring triggers alerts on variance collapse or mode collapse.
- Training artifacts are fully reproducible (data snapshot + decoder + config).

---

## Milestone 7 — Temporal JEPA & Causal World Models

**Intent:** Learn arrow-of-time reasoning from multi-visit sequences.

**Strategic Justification:** Longitudinal sequences enable causal reasoning: foundation → framing → finish is irreversible structure.

**Deliverables**
- `src/transformation_portal/models/vjepa/`
  - temporal JEPA variant trained on ordered multi-visit sequences
- Object persistence training:
  - penalize loss of persistent objects under occlusion or modification

**Acceptance Gates**
- Model improves prediction across time steps without regressing tail diversity.
- Temporal consistency metrics improve on defined eval suites.

---

# Phase III — The “No BS” Validation + Production Layer

## Milestone 8 — Vision-Language Grounding

**Intent:** Fuse visual spatial representations with construction documentation semantics.

**Strategic Justification:** Language-only models are “wordsmiths in the dark.” Pair visuals with floorplans/specs so the system understands terms like “load-bearing wall”.

**Deliverables**
- VL-JEPA:
  - multimodal retrieval: “Find living rooms with south-facing glass walls”
  - schema for pairing docs ↔ scenes ↔ visits

**Acceptance Gates**
- Retrieval tasks are measurable, versioned, and regression-tested.

---

## Milestone 9 — Physics & Consistency Loop

**Intent:** Prevent geometric/physical hallucinations.

**Strategic Justification:** A simulator cannot violate basic constraints (mass, structure, permanence). Validate via rendering and optional physics.

**Deliverables**
- Consistency harness:
  - re-render from novel angles to detect drift/disappearance
- Simulation injection (optional):
  - physics engine checks for static stability support

**Acceptance Gates**
- Consistency checks block promotion of broken assets/models.
- Physics checks produce explainable failures (not silent rejection).

---

## Milestone 10 — Productionization (Secure Service Layer)

**Intent:** Provide enterprise query access without exposing raw moat data.

**Strategic Justification:** This is the moat. Clients query the world model; they do not get the archive.

**Deliverables**
- World Model Query API:
  - change detection
  - future state prediction
  - constrained style transfer
- Compliance & Safety:
  - PII redaction (faces, plates)
  - policy enforcement for outputs and access

**Acceptance Gates**
- API returns derived intelligence only (no raw asset exfiltration).
- Redaction is validated automatically and continuously.

---

## Immediate Thin Slice Execution (Non-Negotiable)

**Start here:** Milestone 2 (High-Fidelity Linear Decode + Training Ingest fork)

**Action**
- Fork the ingestion pipeline immediately.
- Create a dedicated “Training Ingest” path that bypasses 8-bit normalization.

**Why**
You cannot train a High Dynamic Range model on Low Dynamic Range data. If you start foundation training on 8-bit inputs, you permanently bake blindness into the initial weights. Fixing this later requires retraining from scratch.

---

## Execution Model (How Work Lands)

- The roadmap (this file) is the canonical intent.
- Implementation lands as a sequence of PRs aligned to milestones.
- Each milestone MUST have:
  - clear deliverables (docs + code)
  - enforceable acceptance gates (tests + validators + benchmarks)
  - lineage tracking (hashes, manifests, configs)

---

## Required Files (Planned)

- `docs/spatial_ai/ROADMAP.md` (this file)
- `docs/spatial_ai/DATA_CONTRACT.md`
- `docs/spatial_ai/SCENE_SCHEMA.md`
- `docs/spatial_ai/DATASET_OPERATIONS.md` (ingestion + catalog ops, policies, snapshots)

---

## Glossary (Minimal)

- **Clean Moat:** A dataset whose provenance and fidelity are protected by enforceable contracts and gates.
- **Model Collapse:** Degeneration of diversity/variance due to training on synthetic/recursive outputs or polluted distributions.
- **World Model:** A predictive system that models state, causality, and physical consistency, not just pixel appearance.
- **JEPA:** Predictive representation learning that avoids pixel-level generation loops and supports grounding.
- **Longitudinal Multi-Visit:** Repeated captures of the same property across time, enabling arrow-of-time learning.

---
