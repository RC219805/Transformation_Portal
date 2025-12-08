# UI Pack (GitHub + Docs)

This pack improves the repo’s “public UI” in two ways:

1) **GitHub landing UX**
- README template
- issue templates + PR template
- contributor guidance

2) **Docs website UI**
- MkDocs Material site under `docs_site/`
- GitHub Pages deployment workflow

## Enable GitHub Pages deployment
In GitHub:
- Settings → Pages
- Source: GitHub Actions

Then push to `main`; the workflow `pages-docs.yml` deploys automatically.

## Local preview
```bash
pip install -r requirements-docs.txt
./scripts/docs/serve_docs.sh
```
