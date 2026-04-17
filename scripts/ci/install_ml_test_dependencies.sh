#!/usr/bin/env bash

set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
ci_requirements_file="${CI_REQUIREMENTS_FILE:-requirements-ci.txt}"
ml_lockfile="${TP_CI_ML_LOCKFILE:-requirements/ml-core-linux.txt}"
install_ci_requirements=1

while (($#)); do
  case "$1" in
    --skip-ci-requirements)
      install_ci_requirements=0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

"${python_bin}" -m pip install --upgrade pip wheel setuptools

if [[ "${install_ci_requirements}" == "1" ]]; then
  "${python_bin}" -m pip install -r "${ci_requirements_file}"
fi

if [[ ! -f "${ml_lockfile}" ]]; then
  echo "Missing ML lockfile: ${ml_lockfile}" >&2
  exit 1
fi

"${python_bin}" -m pip install -r "${ml_lockfile}"
"${python_bin}" -m pip install -e . --no-deps
"${python_bin}" -m pip check

"${python_bin}" -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
