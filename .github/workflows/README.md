# Transformation Portal Repository

This repository contains tools, scripts, and workflows for managing LUTs, aerial image enhancements, and Montecito manifest generation.

---

## GitHub Actions Workflows

The repository includes multiple CI/CD and automation workflows to ensure code quality, security, and productivity.

### 1. `python-app.yml`
**Purpose:** Main CI workflow for Python testing and linting.
**Triggers:** `push` and `pull_request` on `main`.
**Features:**
- Currently runs on Python 3.11 only (to conserve CI resources).
- Lean CPU-only dependency installation (`requirements-ci.txt`) for fast CI.
- Linting via `flake8` (critical errors only).
- Unit testing and end-to-end tests with `pytest`.
- Montecito manifest generation with artifact upload.
- Test PyPI deployment on main branch pushes for validation.
- Comprehensive cleanup job to prevent disk space issues.

**Note:** Other workflows like `pylint.yml` use the full multi-Python matrix (3.11–3.12) for cross-version consistency testing.

### 2. `submit-pypi.yml`
**Purpose:** Package building and distribution to PyPI and Test PyPI.
**Triggers:**
- Version tags (e.g., `v0.1.0`) for production PyPI.
- Manual workflow dispatch for Test PyPI uploads.

**Features:**
- Builds both wheel and source distributions.
- Comprehensive distribution validation with `twine check`.
- Package content verification to ensure correct structure.
- Test PyPI uploads for validation before production release.
- Production PyPI uploads triggered by version tags only.
- Robust cleanup to prevent disk space issues.
- Separate jobs for build, test upload, and production upload.

**Usage:**
- **Production Release:** Create and push a version tag (e.g., `git tag v0.1.0 && git push origin v0.1.0`)
- **Test PyPI Upload:** Manually trigger workflow with `test_pypi` option enabled
- Uses PyPI Trusted Publishing (OIDC) via `pypa/gh-action-pypi-publish`
- Requires configured GitHub environments (`pypi`, `testpypi`) and matching trusted publisher setup on PyPI/TestPyPI

### 3. `pylint.yml`
**Purpose:** Static code analysis using `pylint`.
**Triggers:** Pull requests affecting `.py` files.
**Features:**
- Multi-Python matrix (3.11–3.12) ensures cross-version consistency.
- Selective linting of changed files to reduce runtime.

### 4. `codeql.yml`
**Purpose:** Security scanning using GitHub CodeQL.
**Features:**
- Automated analysis for security vulnerabilities.
- Runs on pushes to main and pull requests.

### 4.1 `secure-install-pilot.yml`
**Purpose:** Advisory validation of a hash-enforced install pilot for the
checked-in layered dependency contract.
**Triggers:** Pull requests affecting dependency-management surfaces.
**Features:**
- Generates hash-enriched pilot lockfiles into an isolated artifact directory.
- Validates those artifacts with `pip install --dry-run --require-hashes`.
- Covers the non-ML checked-in layered locks only.
- Stays non-blocking while the team evaluates maintenance cost and CI noise.

### 4.2 `frontdoor-deployment-gate.yml`
**Purpose:** Manual predeploy validation for any shared frontdoor rollout that is about to become internet-reachable.
**Triggers:** `workflow_dispatch` only.
**Features:**
- Targets Cloudflare Worker frontdoor rollouts and the legacy Cloudflare-in-front-of-Vercel posture for the managed front door.
- Verifies the public hostname is Cloudflare Access protected and not serving the real DNA shell unauthenticated.
- Verifies the Cloudflare Worker or Vercel deployment URL is protected and not serving the real DNA shell unauthenticated.
- Verifies FastAPI is either non-public by explicit operator attestation or does not expose healthy unauthenticated `/ready` or `/healthz`.
- Runs `make test-frontdoor-contract`, `tests/validation/test_frontdoor_deployment_gate.py`, and `make test-orchestrator-contract` before the live posture probe.
- Uses GitHub environments `frontdoor-staging` and `frontdoor-production` for reviewer-gated rollout approval.

### 5. AI Advisory Workflows

The repository includes three AI-powered advisory workflows that provide intelligent suggestions without blocking PR merges. All workflows follow a hardened pattern with timeout bounds and failure visibility.

**Pattern Documentation:** See `AI_WORKFLOW_PATTERN.md` for the canonical implementation pattern.
**Status Report:** See `AI_WORKFLOWS_HARDENING_STATUS.md` for architectural assessment.

#### 5.1 `ai-code-review.yml` (AI Code Review)
**Purpose:** Automatically reviews code changes in pull requests and provides actionable feedback.
**Status:** Production-ready, non-blocking advisory.

**Features:**
- Triggered on pull request opened/synchronize/reopened events.
- Reviews relevant code files (`.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.yml`, `.yaml`, `.json`, `.md`).
- Uses OpenAI `gpt-4o-mini` model with retry logic for rate-limit handling.
- Posts review comments with code quality assessment, bug detection, security concerns, and best practices when OpenAI returns real review content.
- Keeps AI-unavailable fallback diagnostics in workflow logs instead of posting fallback PR comments.
- Non-blocking: continues even if AI service fails.
- Timeout-bounded: 4-minute step timeout, 10-minute job timeout.
- Requires `OPENAI_API_KEY` in repository secrets.

#### 5.2 `summary.yml` (AI Issue and PR Summarization)
**Purpose:** Automatically generates concise summaries of GitHub issues, pull requests, and comments.
**Status:** Production-ready, non-blocking advisory.

**Features:**
- Triggered on `issues`, `pull_request`, and `issue_comment` events.
- Skips bot-authored `issue_comment` and `issues` events (deployment bots such as
  `cloudflare-workers-and-pages[bot]` and `vercel[bot]` comment on every deploy),
  plus Copilot-authored `pull_request` events.
- Uses OpenAI `gpt-4o-mini` model to generate neutral, concise summaries.
- Retries HTTP 429, transient HTTP 5xx, and network errors up to 3 total attempts.
- Posts successful AI-generated summaries as comments on the issue or pull request.
- Graceful handling when API key is missing or AI calls fail; diagnostic fallbacks stay in logs.
- Non-blocking: continues even if AI service fails.
- Timeout-bounded: 4-minute step timeout, 10-minute job timeout.
- Requires `OPENAI_API_KEY` in repository secrets.

#### 5.3 `smart-issue-management.yml` (AI Issue Triage)
**Purpose:** Automatically classifies and labels issues and pull requests for intelligent triage.
**Status:** Production-ready, non-blocking advisory.

**Features:**
- Triggered on issue/PR opened/reopened/labeled/unlabeled events.
- Uses OpenAI `gpt-4o-mini` model to analyze and classify items.
- Automatically applies labels based on category, priority, and content analysis.
- Posts AI analysis comment with suggested classification.
- Performs duplicate detection for issues.
- Non-blocking: continues even if AI service fails.
- Timeout-bounded: 4-minute step timeout, 10-minute job timeout.
- Requires `OPENAI_API_KEY` in repository secrets.

**AI Workflows Architecture:**
All three workflows implement a hardened pattern with:
- **Non-blocking behavior**: `continue-on-error: true`; expected AI/service failures emit warnings and typically exit 0, while hard infrastructure errors may still exit non-zero
- **Timeout bounds**: 4-minute step timeout for AI calls, 10-minute job timeout
- **Failure visibility**: `::warning::` emission in Python exception handlers and shell failure steps
- **Retry logic**: up to 6 attempts with exponential backoff in `ai-code-review.yml` and `smart-issue-management.yml`; up to 3 attempts with bounded backoff in `summary.yml`
- **Concurrency control**: Cancel outdated runs to reduce CI costs

---

## GitHub Copilot Firewall Configuration

The repository includes a `copilot-firewall.yml` configuration file that specifies allowed external URLs and hosts for GitHub Copilot agents during execution.

**Configuration file:** `.github/copilot-firewall.yml`

**Allowed domains include:**
- Python Package Index (PyPI) for dependency installation
- PyTorch download servers for ML/AI dependencies
- GitHub resources for repository access
- npm registry for managed frontdoor dependency installation
- Hugging Face for ML models and datasets
- OpenAI API for AI summarization
- Common CDNs and NVIDIA toolkit for GPU support

This configuration ensures that Copilot agents can access necessary external resources while maintaining security through explicit allowlisting.

---

## Unit Tests

Unit tests are provided for:

- `_kmeans` – clustering reproducibility.
- `_cluster_stats` – cluster statistics correctness.
- `assign_materials` – assignment logic.
- `_soft_mask` – Gaussian blending of masks.
- `enhance_aerial` – end-to-end test using small sample images.

Run tests locally:

```bash
pip install -r requirements-ci.txt
pytest -v tests/
