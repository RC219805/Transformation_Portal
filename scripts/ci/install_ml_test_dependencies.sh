#!/usr/bin/env bash

set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
ci_requirements_file="${CI_REQUIREMENTS_FILE:-requirements-ci.txt}"
ml_lockfile="${TP_CI_ML_LOCKFILE:-requirements/ml-core-linux.txt}"
raw_requirements_file="${TP_CI_ML_RAW_REQUIREMENTS_FILE:-requirements/ml-raw.in}"
install_ci_requirements=1
install_rawpy=0

while (($#)); do
  case "$1" in
    --skip-ci-requirements)
      install_ci_requirements=0
      ;;
    --include-rawpy)
      install_rawpy=1
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

# The base lane installs GUI OpenCV wheels, while the governed ML lockfile
# owns the headless cv2 provider. Remove all cv2 distributions before the ML
# install so the environment ends with a single deterministic provider.
"${python_bin}" -m pip uninstall -y \
  opencv-python \
  opencv-contrib-python \
  opencv-python-headless \
  opencv-contrib-python-headless >/dev/null 2>&1 || true

"${python_bin}" -m pip install -r "${ml_lockfile}"
if [[ "${install_rawpy}" == "1" ]]; then
  if [[ ! -f "${raw_requirements_file}" ]]; then
    echo "Missing raw requirements file: ${raw_requirements_file}" >&2
    exit 1
  fi
  "${python_bin}" -m pip install -r "${raw_requirements_file}"
fi
"${python_bin}" -m pip install -e . --no-deps
"${python_bin}" -m pip check

"${python_bin}" -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
