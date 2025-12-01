# Semi-Supervised and unsupervised QSAR API (based on FastAPI Production Starter v2.0.0)



## Features

- **QSAR API** endpoints for model metadata & batch prediction
- **FastAPI** (Python 3.12) hardened production setup
- **UV dependency management** for deterministic, fast installs
- **Data pipeline placeholder** (raw -> processed -> predictions)
- **Prometheus metrics** and health / readiness endpoints
- **Optional Sentry** integration
- **Optional PostgreSQL** if persistence needed later
- **GitHub Actions** CI/CD + container publishing
- **Testing & Linting** via pytest + ruff


### Deploy to Render
later
## CI/CD Overview

- `.github/workflows/ci.yml`: lint + tests.
- `.github/workflows/release.yml`: builds & publishes Docker image on version tags (`v*`).
- Uses `GITHUB_TOKEN` to push to GHCR (no extra secrets needed).

## Quick Start

```bash
git clone (url mazal mapushit)
cd mini
cp .env.example .env

# Install dependencies with UV (ida mkaanch uv dir pip install -r requirements.txt w 3awdha m3a requirements dev)
uv sync

# Run locally (no database)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or with PostgreSQL
make docker-compose-up
```

### Alternative: Traditional pip setup

```bash
# If you prefer pip (legacy support)
pip install -r requirements.txt
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

```
┌─────────┐      ┌──────────────┐      ┌──────────┐
│ Client  │─────>│   FastAPI    │─────>│ Database │
└─────────┘      │   + Metrics  │      │(optional)│
                 │   + Sentry   │      └──────────┘
                 └──────────────┘
                        │
                        v
                 ┌──────────────┐
                 │  Prometheus  │
                 │   /metrics   │
                 └──────────────┘
```
##hadod related to the template likan
<!-- 
## Configuration

| Variable       | Description                    | Default                     |
| -------------- | ------------------------------ | --------------------------- |
| `APP_NAME`     | Application name               | FastAPI Template            |
| `APP_VERSION`  | Application version            | 0.1.0                       |
| `ENVIRONMENT`  | Environment (dev/staging/prod) | development                 |
| `DEBUG`        | Debug mode                     | True                        |
| `LOG_LEVEL`    | Logging level                  | INFO                        |
| `DATABASE_URL` | Database connection string     | (optional)                  |
| `SECRET_KEY`   | Secret key for security        | (generated if not provided) |
| `CORS_ORIGINS` | CORS allowed origins           | (empty for security)        |
| `SENTRY_DSN`   | Sentry DSN for error tracking  | (optional)                  |
| `HOST`         | Server host                    | 0.0.0.0                     |
| `PORT`         | Server port                    | 8000                        | -->

## Secret Configuration

The template runs out of the box with no secrets. You can optionally add these for production:

- **`SECRET_KEY`** — Secret used by the application for signing and security. If not provided, a random key is generated at startup in development/test.
- **`DATABASE_URL`** — PostgreSQL connection string. When set, the app uses PostgreSQL; when unset, the app runs without a database and the readiness endpoint will report "not configured" for the database.
- **`SENTRY_DSN`** — When set, Sentry SDK is initialized in the application for error reporting; when empty, Sentry is skipped.

Notes:

- **Workflows do not require any secrets.** CI always runs on pushes and pull requests. The release workflow uses the built‑in `GITHUB_TOKEN` and only pushes images from this repository (guarded by repository owner check). Forks will automatically skip the publishing step.

## Database Options

### Without Database (Minimal)

Leave `DATABASE_URL` empty in `.env` for a minimal API without database dependencies. The app will operate without persistence; `/health/ready` will still return 200 with `database: "not configured"`.

### With PostgreSQL

Set `DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/dbname` in `.env`.

## Development

```bash
# Install dependencies with UV (recommended)
uv sync

# Run development server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
uv run pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# Lint code
uv run ruff check app/ tests/
uv run ruff format --check app/ tests/

# Format code
uv run ruff format app/ tests/

# Build Docker image
make docker-build

# Run with Docker Compose (includes PostgreSQL)
make docker-compose-up
```

### Legacy pip commands (still supported)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# Lint code
ruff check app/ tests/
ruff format --check app/ tests/
```

## Observability

### Prometheus Metrics

- Endpoint: `/metrics`
- Metrics: HTTP requests, duration, status codes
- Integration: `prometheus-fastapi-instrumentator`

### Sentry Integration

- Automatic error tracking when `SENTRY_DSN` is provided
- Performance monitoring
- Skipped automatically when `SENTRY_DSN` is empty

### Health Checks

- `/health` - Basic health check
- `/health/ready` - Readiness check; reports database status and succeeds whether the database is configured or not

## Project Structure

```
app/
├── main.py                  # FastAPI application
├── config.py                # Settings
├── api/routes/health.py     # Health & readiness endpoints
├── api/routes/qsar.py       # QSAR model info + prediction endpoints
├── core/database.py         # Optional async DB (unused by QSAR now)
├── core/metrics.py          # Prometheus instrumentation
├── services/model.py        # Placeholder  QSAR model service
data/
├── raw/features.csv         # Example raw descriptor table
├── processed/               # Processed features / model artifacts
└── predictions/             # Generated prediction outputs
pipeline/run_pipeline.py     # Placeholder data processing + prediction script
notebooks/                   # Jupyter notebooks (exploration, analysis)
tests/                       # Pytest suite
```

## QSAR Endpoints

| Endpoint           | Method | Description                                 |
| ------------------ | ------ | ------------------------------------------- |
| `/qsar/model-info` | GET    | Model metadata (name, version, descriptors) |
| `/qsar/predict`    | POST   | Batch predictions over feature vectors      |

Example batch prediction request:

```json
{
  "items": [{ "features": [0.12, 1.5, 3.2] }, { "features": [0.34, 1.2, 2.8] }]
}
```

Response (placeholder model):

```json
{
  "model": {
    "name": "dummy-semi-supervised-qsar",
    "version": "0.0.1",
    "semi_supervised": true,
    "descriptors": 0,
    "notes": "Placeholder QSAR model. Replace with real implementation."
  },
  "predictions": [
    { "index": 0, "predicted": 0.5201 },
    { "index": 1, "predicted": 0.4307 }
  ]
}
```


For security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## Pipeline

Run the placeholder pipeline (normalizes raw features and produces predictions): NB: hada ghir placeholder mbead nbda n'implement mor mandaw nsyiw  b notebooks

```bash
make pipeline-run
```

Artifacts will appear under `data/processed/` and `data/predictions/`.

