# Disabled workflow archive

`python-app.yml` is retained as historical configuration outside GitHub's active
`.github/workflows/` directory. It has obsolete floating actions, dependency
installation, and publication behavior. Do not copy it back as a template.

Current execution authority is [build.yml](../workflows/build.yml),
[submit-pypi.yml](../workflows/submit-pypi.yml), and the
[workflow matrix](../../docs/ci/WORKFLOW_MATRIX.md).
