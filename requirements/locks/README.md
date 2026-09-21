# Target-owned optional runtime locks

The editorial lane is independent of the six generic locks and the Python 3.11
ML/DA3 lanes. Its owner is `scripts/setup/editorial_runtime.py`; do not add this
lock to the generic lock publication transaction or hand-edit generated files.

## Editorial support and installation

`editorial-darwin-arm64-py312.txt` and its companion JSON describe the full
14-distribution editorial closure for native Apple Silicon, macOS 14+, and
CPython 3.12. The JSON binds exact source/lock digests, shared core pins, compiler
versions, and immutable PyPI wheel URLs/hashes. The installer is standard-library
Python; it does not use ambient pip for runtime installation.

From the repository root, using a trusted Python 3.12 builder:

```bash
.venv/bin/python -I scripts/setup/editorial_runtime.py install
.venv/bin/python -I scripts/setup/editorial_runtime.py check
.venv/bin/python -I scripts/setup/editorial_runtime.py run -- --config /absolute/private/editorial.yml -vv
```

See the [editorial workflow](../../docs/guides/AD_EDITORIAL_POST_PIPELINE.md) for
configuration, retained generation behavior, optional RAW smoke, and acceptance
limits. Runtime files remain under the ignored `.runtime/editorial/` directory.
Do not install this target lock into the core environment.

## Regeneration

`requirements/editorial.in` declares editorial dependencies and constrains shared
versions with `requirements/all.txt`. A dedicated trusted compiler environment
must contain exactly pip 26.2.1, pip-tools 7.6.1, and Click 8.4.2. Create it outside
the repository, without modifying core or optional runtimes:

```bash
python3.12 -m venv /private/tmp/tp-editorial-compiler
/private/tmp/tp-editorial-compiler/bin/python -m pip install \
  --disable-pip-version-check --only-binary=:all: --index-url https://pypi.org/simple \
  'pip==26.2.1' 'pip-tools==7.6.1' 'click==8.4.2'
/private/tmp/tp-editorial-compiler/bin/python -I scripts/setup/editorial_runtime.py lock
.venv/bin/python -I scripts/setup/editorial_runtime.py check --contract-only
.venv/bin/pytest tests/test_editorial_runtime.py -q
```

Use a new owned compiler directory if that example path already exists. The
compiler bootstrap is a maintainer toolchain step, not the runtime's hashed
installation path. Review toolchain and dependency rotations independently.

The pinned compiler resolves the complete native closure; pinned pip then
selects native binary artifacts in a dry-run report. The generator records only
those wheels' hashes and immutable PyPI URLs. This target-specific lock does not
claim multi-platform wheel coverage. Marker decisions are evaluated on the
admitted native host and the final lock contains marker-free exact pins.

Review both generated files together and rerun a staged native installation,
`pip check`, the automatic real encoding smoke, and a representative owned RAW
fixture before accepting a rotation. The installer downloads from the recorded
HTTPS `files.pythonhosted.org` URLs only, rejects redirects, bounds each artifact,
verifies hashes, and installs from its local wheelhouse with `--no-index`,
`--no-deps`, `--require-hashes`, and `--only-binary=:all:`. Check/run also verify
installed payload bytes against retained authenticated wheels before execution.

`check --contract-only` is a portable source-integrity gate, not a network
freshness check or native runtime acceptance. Changes to editorial inputs,
shared closure pins, or either generated file fail this gate until regenerated.
Unrelated generic dependency changes do not force editorial lock churn.
