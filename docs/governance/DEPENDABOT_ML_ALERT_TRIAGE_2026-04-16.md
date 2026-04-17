# Dependabot ML Alert Triage (2026-04-16)

## Scope

This triage record covers the 12 open Dependabot alerts that reduce to 4 advisories repeated across 3 target-owned ML lockfiles:

- `requirements/ml-core-linux.txt`
- `requirements/ml-core-darwin-arm64.txt`
- `requirements/ml-core-darwin-x86_64.txt`

Supported remediation targets:

- `ml-core-darwin-arm64.txt`

Frozen/unsupported lane:

- `ml-core-linux.txt`
- `ml-core-darwin-x86_64.txt`

## Advisory Mapping

| Alert | Advisory | Manifest | Disposition | Evidence / action |
|-------|----------|----------|-------------|-------------------|
| `#129` | PyTorch `torch.load` RCE (`GHSA-53q9-r3pm-6pq6`) | `requirements/ml-core-linux.txt` | Dismiss `not_used` | Linux x86_64 is a frozen unsupported historical ML lane |
| `#143` | PyTorch `torch.load` RCE (`GHSA-53q9-r3pm-6pq6`) | `requirements/ml-core-darwin-arm64.txt` | Remediate | Rotate Darwin arm64 lock/input to `torch==2.8.0`, `torchvision==0.23.0` |
| `#146` | PyTorch `torch.load` RCE (`GHSA-53q9-r3pm-6pq6`) | `requirements/ml-core-darwin-x86_64.txt` | Dismiss `not_used` | Darwin x86_64 is a frozen unsupported ML lane |
| `#128` | PyTorch improper resource shutdown (`GHSA-887c-mr87-cxwp`) | `requirements/ml-core-linux.txt` | Dismiss `not_used` | Linux x86_64 is a frozen unsupported historical ML lane |
| `#142` | PyTorch improper resource shutdown (`GHSA-887c-mr87-cxwp`) | `requirements/ml-core-darwin-arm64.txt` | Remediate | GitHub advisory metadata on 2026-04-16 marks `torch<=2.7.1` as vulnerable, so supported lanes move to the first patched release line: `torch==2.8.0`, `torchvision==0.23.0` |
| `#145` | PyTorch improper resource shutdown (`GHSA-887c-mr87-cxwp`) | `requirements/ml-core-darwin-x86_64.txt` | Dismiss `not_used` | Darwin x86_64 is a frozen unsupported ML lane |
| `#127` | PyTorch local DoS (`GHSA-3749-ghw9-m3mg`) | `requirements/ml-core-linux.txt` | Dismiss `not_used` | Linux x86_64 is a frozen unsupported historical ML lane |
| `#141` | PyTorch local DoS (`GHSA-3749-ghw9-m3mg`) | `requirements/ml-core-darwin-arm64.txt` | Remediate | Closed by the same Darwin arm64 torch uplift |
| `#144` | PyTorch local DoS (`GHSA-3749-ghw9-m3mg`) | `requirements/ml-core-darwin-x86_64.txt` | Dismiss `not_used` | Darwin x86_64 is a frozen unsupported ML lane |
| `#149` | `transformers.Trainer` RCE (`GHSA-69w3-r845-3855`) | `requirements/ml-core-linux.txt` | Dismiss `not_used` | Managed inference paths do not use the vulnerable training/resume flow |
| `#147` | `transformers.Trainer` RCE (`GHSA-69w3-r845-3855`) | `requirements/ml-core-darwin-arm64.txt` | Dismiss `not_used` | Managed inference paths do not use the vulnerable training/resume flow |
| `#148` | `transformers.Trainer` RCE (`GHSA-69w3-r845-3855`) | `requirements/ml-core-darwin-x86_64.txt` | Dismiss `not_used` | Frozen unsupported lane, plus non-reachable training path |

## Trainer Reachability Evidence

Repo search used for the `transformers` alert dismissal:

```bash
rg -n '\bTrainer\b|_load_rng_state|Seq2SeqTrainer|TrainingArguments|resume_from_checkpoint' src tests scripts
```

Observed hits are limited to repository-local RL trainer classes and a streaming checkpoint helper. There are no Hugging Face `Trainer`, `Seq2SeqTrainer`, `TrainingArguments`, `_load_rng_state`, or training-resume imports in supported inference paths.

## Managed Checkpoint Trust Boundary

Managed/orchestrator/server entrypoints keep the `sam2_checkpoint_path` flag, but they now reject arbitrary checkpoint overrides unless the path is:

1. repo-controlled under `./models/sam2/` or `./checkpoints/`, or
2. a file whose SHA-256 matches the governed SAM2 checkpoint manifest.

Pinned Hugging Face revisions remain mandatory where repo-backed loading is used.

## Supported-Lane Audit Runbook

### Darwin arm64 authoritative audit

Run on a native Apple Silicon macOS host:

```bash
python -m pip install -r requirements/security.txt
pip-audit --ignore-vuln CVE-2026-4539 -r requirements/ml-core-darwin-arm64.txt --desc
```

## Sources

- GitHub Dependabot repository alert API for `RC219805/Transformation_Portal` queried on 2026-04-16
- [PyTorch previous versions matrix](https://pytorch.org/get-started/previous-versions/)
- [PyTorch RCE advisory GHSA-53q9-r3pm-6pq6](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)
- [PyTorch local DoS advisory GHSA-3749-ghw9-m3mg](https://github.com/advisories/GHSA-3749-ghw9-m3mg)
- [PyTorch improper resource shutdown advisory GHSA-887c-mr87-cxwp](https://github.com/advisories/GHSA-887c-mr87-cxwp)
- [Hugging Face `Trainer` advisory GHSA-69w3-r845-3855](https://github.com/advisories/GHSA-69w3-r845-3855)
- [Hugging Face fix commit `03c8082`](https://github.com/huggingface/transformers/commit/03c8082ba4594c9b8d6fe190ca9bed0e5f8ca396)
