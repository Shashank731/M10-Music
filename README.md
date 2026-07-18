# Music Recommendation System

Productionized music recommendation app built from the existing notebooks in
`notebooks/`.

The notebooks remain the source of truth for the modeling approach:

- Content-based: song metadata preprocessing, text embeddings, audio feature
  scaling, final weighted embeddings, FAISS similarity search.
- Collaborative: user-track listening history, sparse user-item matrix, ALS
  recommendation path with a runtime fallback when `implicit` is unavailable.

## Project Layout

```text
configs/                  YAML configuration
src/music_recommendation/  ML package: pipeline, recommenders, vector store
backend/                  FastAPI serving app
frontend/                 React client
artifacts/                Trained model/index artifacts
mlruns/                   MLflow tracking output
docker/                   Dockerfiles
tests/                    Unit/API tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Train Artifacts

The backend does not train models. Run these commands first:

```bash
python3 -m music_recommendation.pipeline.content_training
python3 -m music_recommendation.pipeline.als_training
```

Artifacts are written to `artifacts/content` and `artifacts/collaborative`.
MLflow runs are logged under `mlruns`.

## Run API

```bash
PYTHONPATH=src:. uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /search?q=<query>`
- `GET /songs/{song_id}`
- `GET /recommend/song/{song_id}`
- `GET /recommend/user/{user_id}`

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE_URL` if the API is not running on `http://localhost:8000`.

## Docker

```bash
docker compose up --build
```

The backend image expects trained artifacts mounted at `./artifacts`.

## Quality

```bash
ruff check src backend tests
pytest -q
```
