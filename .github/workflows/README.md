# Transformation Portal Repository

This repository contains tools, scripts, and workflows for managing LUTs, aerial image enhancements, and Montecito manifest generation.

---

## GitHub Actions Workflows

The repository includes multiple CI/CD and automation workflows to ensure code quality, security, and productivity.

### 1. `python-app.yml`
**Purpose:** Main CI workflow for Python testing and linting.  
**Triggers:** `push` and `pull_request` on `main`.  
**Features:**
- Multi-Python testing matrix (3.10–3.12).  
- Lean CPU-only dependency installation (`requirements-ci.txt`) for fast CI.  
- Linting via `flake8` (critical errors only).  
- Unit testing and end-to-end tests with `pytest`.  
- Montecito manifest generation with artifact upload.  

### 2. `pylint.yml`
**Purpose:** Static code analysis using `pylint`.  
**Triggers:** Pull requests affecting `.py` files.  
**Features:**
- Multi-Python matrix (3.10–3.12) ensures cross-version consistency.  
- Selective linting of changed files to reduce runtime.  

### 3. `codeql.yml`
**Purpose:** Security scanning using GitHub CodeQL.  
**Features:**
- Automated analysis for security vulnerabilities.  
- Runs on pushes to main and pull requests.  

### 4. `summary.yml` (AI Issue and PR Review Summarization)
**Purpose:** Automatically generates a summary of newly opened GitHub issues, pull requests, and pull request reviews.  
**Status:** Fully functional with OpenAI API integration.

**Features:**
- Triggered on `issues.opened`, `issues.edited`, `pull_request`, `pull_request_review`, and `issue_comment`.  
- Uses OpenAI `gpt-4o-mini` model to summarize issue/PR/review content.  
- Posts the summary as a comment on the issue or pull request.  
- Includes graceful fallback if API call fails.  
- Requires `OPENAI_API_KEY` in repository secrets.

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
