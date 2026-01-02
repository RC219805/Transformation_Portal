# PyPI Trusted Publishing Setup Guide

## What Changed

The `submit-pypi.yml` workflow now uses **Trusted Publishing** (OpenID Connect) instead of API tokens. This is the modern, more secure approach recommended by PyPI.

## Why This Matters

- ✅ **No secrets to manage** - No `PYPI_API_TOKEN` needed
- ✅ **More secure** - Uses GitHub's OIDC provider for authentication
- ✅ **Automatic rotation** - Tokens are short-lived and auto-generated per run
- ✅ **Audit trail** - PyPI knows exactly which GitHub workflow published each release

## Setup Instructions

### Step 1: Configure PyPI (One-time setup)

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/publishing/)
2. Scroll to "Publishing" section
3. Click "Add a new publisher"
4. Fill in the form:
   - **PyPI Project Name**: `transformation-portal`
   - **Owner**: `RC219805`
   - **Repository name**: `Transformation_Portal`
   - **Workflow name**: `submit-pypi.yml`
   - **Environment name**: (leave empty)

5. Click "Add"

### Step 2: Configure Test PyPI (Optional, for testing)

1. Go to [Test PyPI Account Settings](https://test.pypi.org/manage/account/publishing/)
2. Follow same steps as above

### Step 3: Test the Workflow

#### Option A: Test with Test PyPI (Recommended first)
```bash
# Trigger manual workflow run
gh workflow run submit-pypi.yml --ref main -f test_pypi=true
```

#### Option B: Full production release
```bash
# Create and push a new tag
git tag -a v2.0.1 -m "Release v2.0.1"
git push origin v2.0.1
```

## Troubleshooting

### Error: "403 Forbidden - Invalid or non-existent authentication"

**Cause**: Trusted Publishing not configured on PyPI.

**Solution**: Complete Step 1 above.

### Error: "Trusted publishing exchange failure"

**Possible causes**:
1. Workflow filename mismatch - Must be exactly `submit-pypi.yml`
2. Repository name mismatch - Check spelling in PyPI settings
3. Missing `id-token: write` permission in workflow

### Error: "Package does not match publisher settings"

**Cause**: PyPI project name doesn't match the package name in `pyproject.toml`.

**Solution**: Verify the package name in `pyproject.toml` matches the PyPI project name exactly.

## Verification

After setup, you can verify the configuration:

1. Check PyPI project settings: https://pypi.org/manage/project/transformation-portal/settings/publishing/
2. Look for "Trusted publishers" section - should show:
   - Owner: RC219805
   - Repository: Transformation_Portal
   - Workflow: submit-pypi.yml

## Migration Notes

### What was removed
- `PYPI_API_TOKEN` secret (no longer needed)
- `TEST_PYPI_API_TOKEN` secret (no longer needed)
- `twine` upload step (replaced with `pypa/gh-action-pypi-publish`)

### What was added
- `id-token: write` permission to both `pypi` and `test-pypi` jobs
- `pypa/gh-action-pypi-publish@release/v1` action

## Security Benefits

1. **Short-lived tokens**: Each workflow run gets a unique, temporary token
2. **No credential storage**: Nothing to leak in repository settings
3. **Provenance**: PyPI records which exact GitHub workflow published each version
4. **Automatic rotation**: No manual token refresh needed

## References

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish)
