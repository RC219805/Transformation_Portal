# Getting Started

## Install (recommended)
Use the curated safe dependency set:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r lux_depth_v2/requirements-repo.txt
```

## Run the service (example)
Start the FastAPI service (exact entrypoint depends on your repo wiring):

```bash
python -m lux_depth_v2.service
```

Typical endpoints:
- `GET /health`
- `POST /v2/process`
- `GET /metrics` (if observability pack enabled)

## Run local docs UI
This UI pack ships with an MkDocs Material site:

```bash
pip install -r requirements-docs.txt
./scripts/docs/serve_docs.sh
```

Open `http://127.0.0.1:8000/`.

