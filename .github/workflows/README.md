# Transformation Portal Workflows

This directory contains GitHub Actions workflows for CI/CD, quality assurance, and automation.

---

## GitHub Actions Workflows

The repository includes multiple CI/CD and automation workflows to ensure code quality, security, and productivity.

### 1. `ci-consolidated.yml` ⭐ (Primary CI/CD)
**Purpose:** Unified CI/CD pipeline with intelligent job orchestration.  
**Triggers:** `push` to `main`/`develop`, `pull_request` to `main`, manual dispatch.  
**Features:**
- **40-60% faster** than previous fragmented workflows
- Intelligent change detection for targeted test runs
- Multi-Python matrix (3.10, 3.11, 3.12)
- Shared caching across jobs
- RAG system validation stage
- Security checks (basicsr CVE-2024-27763 mitigation)
- ML tests with CPU-only PyTorch for efficiency
- **Phase 2 feature flags** for CLIP/Lighting Detection (NEW)
- **Optional performance benchmarking** with regression checks (NEW)

**Stages:**
1. Setup & Change Detection
2. Lint & Quality Checks
3. Core Tests (no ML dependencies)
4. ML Tests (requires PyTorch, conditional Phase 2 features)
4.2. Phase 2 Performance Benchmark (optional)
5. Lux Depth V2 Module Tests
6. RAG System Validation
7. Build Artifacts
8. Generate Montecito Manifest
9. Pipeline Summary

**Phase 2 Feature Flags:**
- `enable_phase2_features` (default: `true`) - Enable CLIP classification and Lighting Detection
- `run_benchmark_regression` (default: `false`) - Run Phase 2 performance benchmarks

**Usage:**
```bash
# Disable Phase 2 features for testing
gh workflow run ci-consolidated.yml -f enable_phase2_features=false

# Run with benchmark regression
gh workflow run ci-consolidated.yml -f run_benchmark_regression=true
```

**Documentation:** See `docs/CI_PHASE2_INTEGRATION.md` for detailed Phase 2 integration guide.

**Note:** This workflow replaces the deprecated `build.yml` and `python-app.yml`.

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
- Requires `PYPI_API_TOKEN` and `TEST_PYPI_API_TOKEN` in repository secrets

### 3. `quality-gate.yml`
**Purpose:** Pre-commit quality checks and auto-formatting.  
**Triggers:** `push` and `pull_request` on `main`.  
**Features:**
- Auto-fixes formatting issues with `autopep8`
- Runs flake8 for critical errors
- Runs pylint (non-blocking)
- Enforces markdown file count limits in root directory

### 4. `codeql.yml`
**Purpose:** Security scanning using GitHub CodeQL.  
**Features:**
- Automated analysis for security vulnerabilities.  
- Runs on pushes to main and pull requests.  

### 5. `summary.yml` (AI Issue and PR Review Summarization)
**Purpose:** Automatically generates a summary of newly opened GitHub issues, pull requests, and pull request reviews.  
**Status:** Fully functional with OpenAI API integration.

**Features:**
- Triggered on `issues.opened`, `issues.edited`, `pull_request`, `pull_request_review`, and `issue_comment`.  
- Uses OpenAI `gpt-4o-mini` model to summarize issue/PR/review content.  
- Posts the summary as a comment on the issue or pull request.  
- Includes graceful fallback if API call fails.  
- Requires `OPENAI_API_KEY` in repository secrets.

### 6. Additional Workflows

| Workflow | Purpose |
|----------|---------|
| `ai-code-review.yml` | AI-powered code review integration |
| `dependency-submission.yml` | Dependency graph submission to GitHub |
| `dependency-update.yml` | Automated dependency updates |
| `performance-monitor.yml` | Performance regression monitoring |
| `pr-context.yml` | PR context enrichment |
| `security-scan.yml` | Additional security scanning |
| `smart-issue-management.yml` | Intelligent issue triaging |
| `trend-dashboard.yml` | Quality trends visualization |

### Deprecated Workflows

The following workflows have been superseded by `ci-consolidated.yml`:
- `build.yml.deprecated` - Legacy build workflow
- `python-app.yml.deprecated` - Legacy Python CI/CD workflow

---

## GitHub Copilot Firewall Configuration

The repository includes a `copilot-firewall.yml` configuration file that specifies allowed external URLs and hosts for GitHub Copilot agents during execution.

**Configuration file:** `.github/copilot-firewall.yml`

**Allowed domains include:**
- Python Package Index (PyPI) for dependency installation
- PyTorch download servers for ML/AI dependencies
- GitHub resources for repository access
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
