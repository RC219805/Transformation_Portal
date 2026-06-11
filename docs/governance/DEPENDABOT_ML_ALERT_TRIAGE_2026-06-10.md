# Dependabot ML Alert Triage (2026-06-10)

## Scope

Live GitHub Dependabot state on 2026-06-10 showed seven open PyTorch alerts
and no open code-scanning or secret-scanning alerts.

```bash
gh api 'repos/RC219805/Transformation_Portal/dependabot/alerts?state=open&per_page=100' \
  --jq '.[] | [.number, .security_advisory.ghsa_id, .security_vulnerability.severity, .dependency.manifest_path, .dependency.package.name, .security_vulnerability.vulnerable_version_range, (.security_vulnerability.first_patched_version.identifier // "none"), .security_advisory.summary] | @tsv'
gh api 'repos/RC219805/Transformation_Portal/code-scanning/alerts?state=open&per_page=100' --jq '[.[]]'
gh api 'repos/RC219805/Transformation_Portal/secret-scanning/alerts?state=open&per_page=100' --jq '[.[]]'
```

## Advisory Mapping

| Alert | Advisory | Manifest | Disposition | Evidence / action |
|-------|----------|----------|-------------|-------------------|
| `#219` | PyTorch `torch.lstm_cell` memory corruption (`GHSA-qfhq-4f3w-5fph`) | `requirements/ml-core-darwin-arm64.txt` | Remediate | Rotate supported Apple Silicon lock/input to `torch==2.12.0`, above the first patched `2.10.0` line. |
| `#218` | PyTorch `torch.jit.script` memory corruption (`GHSA-rrmf-rvhw-rf47`) | `requirements/ml-core-darwin-arm64.txt` | Dismiss `not_used` | No patched PyTorch release is available on the PyTorch CPU index; repository search shows no `torch.jit.script` use. Supported lock still rotates to latest available `torch==2.12.0`. |
| `#217` | PyTorch `unpack_sequence` memory corruption (`GHSA-vgrw-7cvw-pwgx`) | `requirements/ml-core-darwin-arm64.txt` | Remediate | Rotate supported Apple Silicon lock/input to `torch==2.12.0`, above the first patched `2.9.1` line. |
| `#216` | PyTorch `torch.lstm_cell` memory corruption (`GHSA-qfhq-4f3w-5fph`) | `requirements/all.txt` | Dismiss `inaccurate` | Current `requirements/all.txt` no longer contains `torch`; package is not in this manifest. |
| `#215` | PyTorch `torch.jit.script` memory corruption (`GHSA-rrmf-rvhw-rf47`) | `requirements/all.txt` | Dismiss `inaccurate` | Current `requirements/all.txt` no longer contains `torch`; package is not in this manifest. |
| `#214` | PyTorch `unpack_sequence` memory corruption (`GHSA-vgrw-7cvw-pwgx`) | `requirements/all.txt` | Dismiss `inaccurate` | Current `requirements/all.txt` no longer contains `torch`; package is not in this manifest. |
| `#213` | PyTorch `torch.jit.script` memory corruption (`GHSA-rrmf-rvhw-rf47`) | `config/fastvlm_runtime_requirements.txt` | Dismiss `not_used` | FastVLM optional runtime rotates to latest available `torch==2.12.0`; repository search shows no `torch.jit.script` use. |

## Reachability Evidence

The unpatched advisory `GHSA-rrmf-rvhw-rf47` is scoped to
`torch.jit.script`. The repository does use `torch.jit.trace` for CoreML
conversion, but that is not the vulnerable API named by the alert.

```bash
rg -n '\b(torch\.jit\.script|jit\.script)\b' src tests scripts config requirements*.txt requirements .github
rg -n '\b(unpack_sequence|lstm_cell|torch\.lstm_cell)\b' src tests scripts config requirements*.txt requirements .github
rg -n '^torch==|^torchvision==' requirements/ml-core-darwin-arm64.txt config/fastvlm_runtime_requirements.txt
rg -n '^torch==' requirements/all.txt
```

Observed result:

- No `torch.jit.script`, `unpack_sequence`, `lstm_cell`, or
  `torch.lstm_cell` hits in the scanned repo paths.
- `requirements/ml-core-darwin-arm64.txt` now pins `torch==2.12.0` and
  `torchvision==0.27.0`.
- `config/fastvlm_runtime_requirements.txt` now pins `torch==2.12.0` and
  `torchvision==0.27.0`.
- `requirements/all.txt` has no `torch` pin.

## Package Availability

```bash
.venv/bin/python -m pip index versions torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip index versions torchvision --index-url https://download.pytorch.org/whl/cpu
```

Observed result on 2026-06-10:

- Latest PyTorch CPU index version: `torch 2.12.0`
- Latest paired torchvision CPU index version: `torchvision 0.27.0`

## Local pip-audit Evidence

```bash
.venv/bin/python -m pip_audit -r requirements/ml-core-darwin-arm64.txt --desc
.venv/bin/python -m pip_audit -r config/fastvlm_runtime_requirements.txt --desc
```

Observed result:

- Both manifests resolve with `torch==2.12.0`.
- `pip-audit` reports one remaining PyTorch finding, `CVE-2025-3000`, with no
  fix version. The finding affects `torch.jit.script`, matching
  `GHSA-rrmf-rvhw-rf47`.
- The repository does not call `torch.jit.script` in runtime/source paths, so
  this advisory is dispositioned as `not_used` until upstream publishes a
  patched PyTorch release.

## Follow-Up

If upstream publishes a patched PyTorch release for
`GHSA-rrmf-rvhw-rf47`, rotate the supported Apple Silicon lock, FastVLM
runtime requirements, pyproject metadata, and torch security baseline again
instead of keeping the `not_used` dismissal.
