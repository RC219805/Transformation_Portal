#!/usr/bin/env bash

set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
ci_requirements_file="${CI_REQUIREMENTS_FILE:-requirements-ci.txt}"
torch_version="${TP_CI_TORCH_VERSION:-2.10.0+cpu}"
torchvision_version="${TP_CI_TORCHVISION_VERSION:-0.25.0+cpu}"
pytorch_index_url="${TP_CI_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
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

"${python_bin}" -m pip install \
  "torch==${torch_version}" \
  "torchvision==${torchvision_version}" \
  --index-url "${pytorch_index_url}"
"${python_bin}" -m pip install -e ".[ml]"

if [[ -f "${ci_requirements_file}" ]]; then
  sklearn_constraint="$(grep -E '^scikit-learn[^#]*' "${ci_requirements_file}" | head -n1 || true)"
  if [[ -n "${sklearn_constraint}" ]]; then
    "${python_bin}" -m pip install --force-reinstall "${sklearn_constraint}"
  fi
fi

"${python_bin}" -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
