# Workflow Parser

A tool to parse and validate GitHub Actions workflow files for common bugs and issues.

## Usage

```bash
python parse_workflows.py
```

The script will automatically scan all workflow files in `.github/workflows/` and report any issues found.

## What It Detects

### Errors (Critical Issues)

1. **Unclosed Conditionals** - Missing `fi` statements in shell scripts
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

2. **Invalid API References** - Potentially invalid OpenAI model names
   ```yaml
   -d '{"model": "gpt-4.1-mini"}'  # ⚠️ Invalid model name
   ```

## Example Output

```
================================================================================
Found 3 issue(s) in workflow files:
================================================================================

[ERROR] .github/workflows/build.yml:49 - Unclosed conditional in job 'lint-and-test': found 1 'if' statements but 0 'fi' statements

[ERROR] .github/workflows/summary.yml:46 - Step output referenced 'steps.generate-summary.outputs' but step id 'generate-summary' not found in job 'summarize-issue'

[WARNING] .github/workflows/build.yml:17 - Job 'lint-and-test' has device matrix [cpu, gpu] but includes 'lint' task which doesn't require multiple devices

================================================================================
Summary: 2 error(s), 1 warning(s), 0 info
================================================================================
```

## Testing

Run the test suite:

```bash
pytest tests/test_parse_workflows.py -v
```

## Fixed Issues

This parser helped identify and fix the following bugs:

1. **build.yml:52** - Added missing `fi` statement to close conditional
2. **summary.yml:19** - Added `id: generate-summary` to step
3. **summary.yml:28** - Changed invalid model "gpt-4.1-mini" to "gpt-4o-mini"
4. **build.yml:17** - Added matrix exclusion to prevent lint running on both cpu/gpu

## Integration

This parser can be integrated into CI/CD pipelines to automatically check workflows:

```yaml
- name: Validate workflows
  run: python parse_workflows.py
```
