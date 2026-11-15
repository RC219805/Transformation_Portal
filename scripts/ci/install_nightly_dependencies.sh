#!/usr/bin/env bash

# All four Nightly lanes and manual install-only qualification use this path.
set -euo pipefail

if (($#)); then
  echo "Usage: $0" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
bash "${script_dir}/install_ml_test_dependencies.sh" --cpu
"${python_bin}" -m pip install memory-profiler psutil pytest-timeout pytest-benchmark pytest-memray
"${python_bin}" -m pip check
"${python_bin}" -c 'import memory_profiler, psutil, pytest_benchmark, pytest_jsonreport, pytest_memray, pytest_timeout; print("Nightly test tooling imports passed")'
