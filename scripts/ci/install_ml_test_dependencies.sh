#!/usr/bin/env bash

set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
ci_requirements_file="${CI_REQUIREMENTS_FILE:-requirements-ci.txt}"
ml_lockfile="${TP_CI_ML_LOCKFILE:-}"
raw_requirements_file="${TP_CI_ML_RAW_REQUIREMENTS_FILE:-requirements/ml-raw.in}"
install_ci_requirements=1
install_rawpy=0
project_installed=0
install_cpu=0

while (($#)); do
  case "$1" in
    --skip-ci-requirements)
      install_ci_requirements=0
      ;;
    --include-rawpy)
      install_rawpy=1
      ;;
    --cpu)
      install_cpu=1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "${install_cpu}" == "1" ]]; then
  if [[ -n "${ml_lockfile}" ]]; then
    echo "--cpu cannot be combined with TP_CI_ML_LOCKFILE" >&2
    exit 2
  fi
  "${python_bin}" -c '
import platform
import sys

if sys.platform != "linux" or platform.machine() != "x86_64":
    raise SystemExit("--cpu requires Linux x86_64")
'
fi

"${python_bin}" -m pip install --upgrade pip wheel setuptools

if [[ "${install_ci_requirements}" == "1" ]]; then
  "${python_bin}" -m pip install -r "${ci_requirements_file}"
fi

# The CI base layer and the governed ML lockfile can select different cv2
# distributions. Remove all OpenCV wheels before the ML install so the
# environment ends with a single deterministic provider.
"${python_bin}" -m pip uninstall -y \
  opencv-python \
  opencv-contrib-python \
  opencv-python-headless \
  opencv-contrib-python-headless >/dev/null 2>&1 || true

if [[ -n "${ml_lockfile}" ]]; then
  if [[ ! -f "${ml_lockfile}" ]]; then
    echo "Missing ML lockfile: ${ml_lockfile}" >&2
    exit 1
  fi
  "${python_bin}" -m pip install -r "${ml_lockfile}"
else
  echo "No TP_CI_ML_LOCKFILE set; installing the governed CI torch baseline before ML core extras."
  # The package extras intentionally expose compatible lower bounds for
  # operators. Hosted golden tests need the repository's audited baseline,
  # otherwise a newly published torch/torchvision minor can change numeric
  # output without any repository change.
  if [[ "${install_cpu}" == "1" ]]; then
    "${python_bin}" -m pip install \
      "torch==2.13.0+cpu" \
      "torchvision==0.28.0+cpu" \
      --index-url https://download.pytorch.org/whl/cpu
  else
    "${python_bin}" -m pip install \
      "torch==2.13.0" \
      "torchvision==0.28.0"
  fi
  "${python_bin}" -m pip install -e ".[ml-core]"
  project_installed=1
fi

if [[ "${install_rawpy}" == "1" ]]; then
  if [[ ! -f "${raw_requirements_file}" ]]; then
    echo "Missing raw requirements file: ${raw_requirements_file}" >&2
    exit 1
  fi
  "${python_bin}" -m pip install -r "${raw_requirements_file}"
fi
if [[ "${project_installed}" == "0" ]]; then
  "${python_bin}" -m pip install -e . --no-deps
fi
"${python_bin}" -m pip check

"${python_bin}" -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"

if [[ "${install_cpu}" == "1" ]]; then
  # Verify after metadata extras: resolution must not replace the CPU pair.
  "${python_bin}" -c '
import torch
import torchvision

if torch.__version__ != "2.13.0+cpu":
    raise SystemExit(f"Expected torch==2.13.0+cpu; got {torch.__version__}")
if torchvision.__version__ != "0.28.0+cpu":
    raise SystemExit(f"Expected torchvision==0.28.0+cpu; got {torchvision.__version__}")
if torch.version.cuda is not None:
    raise SystemExit(f"Expected a CPU-only torch build; CUDA is {torch.version.cuda}")
print("Verified governed CPU torch/torchvision pair")
'
fi
