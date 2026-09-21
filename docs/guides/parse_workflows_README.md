# Workflow Parser

A tool to parse and validate GitHub Actions workflow files for common bugs and issues.

## Usage

```bash
.venv/bin/python scripts/validation/parse_workflows.py
```

Run from the repository root with the core environment installed. The script
scans `.github/workflows/` and reports findings. Use `--workflow-dir <directory>`
for fixtures and `--format json` for structured diagnostics. The current CLI
prints progress lines before the JSON diagnostics; stdout is not a standalone
JSON document. Errors return exit status 1; warnings and informational skips
return 0.

Shell checking requires `/bin/bash` and `/bin/sh`. It passes supported shell
bodies to the matching interpreter with `-n`, without executing workflow
commands or loading ambient shell startup configuration. Step `shell` overrides
job defaults, which override workflow defaults. Known Linux/macOS runners use
Bash by default; container jobs use sh. Explicit Python/PowerShell, unresolved
runner, container, or shell expressions, and unsupported custom shell templates are reported
as skipped. This is a local static check: it does not evaluate Actions expressions,
validate embedded Python, or prove runner behavior or remote API availability.

## What It Detects

### Errors (Critical Issues)

1. **Shell Syntax Errors** - Including missing `fi` statements in Bash/sh scripts;
   heredocs and quoted strings are parsed as shell syntax rather than counted as
   conditional keywords.
   ```yaml
   run: |
     if [ -z "$VAR" ]; then
       echo "missing fi"  # ❌ Missing fi
   ```

2. **Missing Step IDs** - Step outputs referenced without corresponding step IDs
   ```yaml
   - name: Generate value
     run: echo "value=test" >> "$GITHUB_OUTPUT"
   - name: Use value
     run: echo ${{ steps.generate.outputs.value }}  # ❌ No id: generate
   ```

3. **Invalid Job Dependencies** - Jobs that depend on non-existent jobs
   ```yaml
   job2:
     needs: nonexistent_job  # ❌ Job doesn't exist
   ```

4. **YAML Syntax Errors** - Malformed YAML that can't be parsed

### Warnings (Optimization Opportunities)

1. **Inefficient Matrix Usage** - Matrix configurations that create unnecessary jobs
   ```yaml
   matrix:
     task: [lint, test]
     device: [cpu, gpu]  # ⚠️ Lint doesn't need both devices
   ```

2. **Legacy Model Hints** - A model name absent from the script's historical
   static list produces an advisory warning. That list cannot establish whether
   a model exists, supports an endpoint, or is available to an account. Do not
   replace a model merely to clear this warning; verify provider documentation
   and the workflow's actual API contract.

## Example Output

```
================================================================================
Found 3 issue(s) in workflow files:
================================================================================

[ERROR] .github/workflows/example.yml:9 - Shell syntax error in job 'lint-and-test', step 1 (bash)
  Context: /bin/bash: line 3: syntax error: unexpected end of file

[ERROR] .github/workflows/summary.yml:46 - Step output referenced 'steps.generate-summary.outputs' but step id 'generate-summary' not found in job 'summarize-issue'

[WARNING] .github/workflows/build.yml:17 - Job 'lint-and-test' has device matrix [cpu, gpu] but includes 'lint' task which doesn't require multiple devices

================================================================================
Summary: 2 error(s), 1 warning(s), 0 info
================================================================================
```

## Testing

Run the actual CLI regression suite:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_parse_workflows_cli.py -v
```

`tests/test_parse_workflows.py` covers the separate package analyzer under
`src/transformation_portal/analyzers`; it does not validate this script's CLI.

## Integration

This parser can be integrated into CI/CD pipelines to automatically check workflows:

```yaml
- name: Validate workflows
  run: python scripts/validation/parse_workflows.py
```
