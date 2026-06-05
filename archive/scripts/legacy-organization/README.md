# Legacy Organization Scripts

This directory retains retired organization helpers, branch-specific cleanup
scripts, and stale duplicate script implementations as historical evidence.
These files are not active operator entrypoints.

Archived branch cleanup, one-off structure generation, context quick-start, and
project extraction scripts use lowercase snake_case names with a `_legacy`
suffix. Keep new active helpers under the governed `scripts/` subdirectories
instead of reviving these paths.

`verify_pr98_legacy.sh` is retained only as PR #98 historical evidence. It
checks obsolete paths such as root `format_utils.py` and must not be restored as
a current validation entrypoint.

`execute_phase_2_extraction_legacy.sh` is retained only as 2024 Materials V3
project-status evidence. It performs branch checkout, pull, stash, and commit
operations and is not a safe current operator workflow.

`process_750_picacho_batch_legacy.py` is retained only as a legacy 750 Picacho
batch wrapper. It targets an obsolete Lux Render API shape and must not be
restored as an active pipeline runner.

Use the governed validators instead:

- `./.auto-organize.sh --check --verbose`
- `python3 scripts/governance/check_script_topology.py --verbose`
- `python3 scripts/governance/check_docs_structure.py --all`
