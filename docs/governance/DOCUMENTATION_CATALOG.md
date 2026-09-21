# Current documentation catalog

The [catalog](documentation_catalog.json) is the current machine-readable
inventory and authority record. The [documentation map](DOCUMENTATION_MAP.md)
remains the human navigation entrypoint. The September 12 inventory and other
dated audits remain unchanged historical evidence.

## What authority means

Every tracked file under `docs/`, plus tracked Markdown, MDX and reStructuredText
elsewhere, has an explicit record. The checker also includes prospective
non-ignored documents in a local worktree so new files cannot silently escape
classification. Downloaded runtimes and ignored build output are outside scope.

Classification is one of `canonical`, `current-support`, `mixed`, `historical`,
or `archive-only`. Review depth is independent:

- `source-reviewed` binds the document and listed implementation/test references
  to their exact SHA256 bytes and a reviewed source baseline.
- `inherited-classification` retains prior classification without claiming a
  current source review.
- `historical-evidence` preserves a dated record; its old commands are not live
  operator instructions.
- `generated-snapshot` identifies machine-generated metadata. The catalog's own
  record has a null document hash to avoid recursive self-hashing and cannot
  authorize retrieval priority.

The `source_commit` is the source baseline used for review; document and
`source_sha256` hashes identify the exact reviewed bytes, including reviewed
changes prepared on top of that baseline. A commit alone does not certify
changed working-tree content. `maintenance_area` names a repository area, not a
newly assigned person. `scope` states capability and acceptance limits.

`source-contract` evidence means source/contract review. `inventory-only`,
`historical-record`, and `generated-metadata` have narrower meanings. None
implies native-runtime, photographic, hosted-service, or production acceptance.
Validation commands are review pointers, never shell instructions executed by
the catalog checker.

## Updating a document

1. Change the existing canonical guide when it already covers the workflow.
   Recheck commands against source, CLI help and the applicable focused tests.
2. Edit its catalog record explicitly when classification, scope, source/test
   references, review status, or successors change. New documents require a
   complete record; no filename heuristic promotes them to maintained authority.
3. Refresh only the paths just reviewed, passing the exact source baseline:

   ```bash
   .venv/bin/python scripts/governance/check_documentation_catalog.py \
     --refresh docs/guides/SETUP_GUIDE.md \
     --source-commit <reviewed-40-character-commit>
   ```

   This updates the document and referenced-source hashes without changing
   classification or review status. Multiple reviewed paths may follow
   `--refresh`. Refresh does not silently classify additions or remove deleted
   records. The complete catalog must validate before atomic publication.
4. Run `make check-documentation-catalog` and
   `make test-documentation-contract`. Keep the catalog update in the same
   change as the guide or its referenced implementation changes.

Review source drift even when prose did not change. The validator rejects
modified source references until affected guidance is explicitly re-reviewed.
Historical corrections also require a document hash refresh, but retain their
historical status. Do not rewrite a dated audit to make its observations current.

## Successors and navigation

`successors` contains cataloged paths; unknown targets and cycles fail. The
record's scope describes the replacement boundary. For example, MaterialsV4
succeeds material evidence and photographic response, not six-map physical PBR.
V5 is an opt-in depth/photographic candidate; V3 remains the production baseline,
and editorial HDR/panorama/ProPhoto acceptance remains independent.

Current navigation sources are explicitly listed. Links to mixed or historical
records require a source/target-specific `historical_navigation` reason, and
the visible link or its row/prose context must include a dated/evidence qualifier
(for example, "prior audit" or "historical snapshot"). A historical pathname
alone cannot supply that visible label. Stale exceptions fail.
The V3 pipeline operations guide is explicitly maintained despite its historical
parent directory; this does not promote neighboring legacy material.

## Validation coverage

`make check-documentation-catalog` checks inventory closure, classifications,
review metadata, safe paths, hashes, source/test references, successor cycles,
maintained local links and historical navigation routing. Maintained Markdown
checks include inline, reference, image and HTML links, heading fragments and
explicit anchors while excluding fenced/indented/inline code and comments.
It does not validate remote URLs, execute Markdown, or interpret RST roles and
dynamic MDX-generated anchors. Historical link debt is retained as evidence.

`make test-documentation-contract` runs curated catalog, RAG and operator-example
regressions. Actual APEX examples are parsed and passed to resolver-only planning
with temporary local inputs. The PBR example executes its real CLI route using
a small image. The workflow parser uses non-executing shell syntax checks and
fixtures for actual syntax failures. RAG examples use supplied chunks and offline retrieval. Editorial
image/ICC/atomic-output contracts have their own focused suite and runtime gate.
These are specific assertions, not comprehensive production acceptance.

Existing placement and changed-root-path validators remain available. Their
green output proves their stated narrower scope. The new catalog gate is
additive to the blocking lightweight CI documentation checks; no required check
name, route, selector, package version, or execution schema is replaced.

## Retrieval modes

The offline RAG entrypoint uses catalog authority metadata alongside relevance.
Operator queries prefer matching source-reviewed current guidance whose
document and source hashes still match. Missing or stale metadata remains
unverified, never implicit current authority. Historical research is available
through explicit `historical` and `all` modes; old evidence is retained.

Catalog bytes and referenced-source bytes participate in cache identity, even
when a reference such as `Makefile` is outside the retrieval corpus. A metadata
change or source change cannot reuse a cache carrying obsolete authority.
See the [RAG quickstart](../../.github/agents/RAG_QUICK_START.md) for runnable
commands. Retrieval rank is not proof of correctness, approval, or readiness.
