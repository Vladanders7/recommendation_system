Feed Recommendation Service (FastAPI + CatBoost)

A baseline recommendation system that returns top-5 personalised post
recommendations for a given user.

The service is built as a production-style API using FastAPI and
CatBoost.

------------------------------------------------------------------------

MODEL

Algorithm: CatBoostClassifier Metric: HitRate@5 ≈ 0.60

Inference is optimised for latency: - Candidate pool caching - Single
model load at startup - In-memory user cache

------------------------------------------------------------------------

PROJECT STRUCTURE

service/ app.py - FastAPI inference service

src/ train.py - training pipeline (baseline)

data/sample/ - small CSV samples for local mode

artifacts/ model.cbm - model weights (NOT committed to GitHub)

------------------------------------------------------------------------

RUNNING THE SERVICE

1)  Create virtual environment

python3 -m venv .venv source .venv/bin/activate

2)  Install dependencies

pip install -r requirements.txt

3)  Place model file

Put trained model weights into:

artifacts/model.cbm

------------------------------------------------------------------------

LOCAL MODE (WITHOUT DATABASE)

By default the service runs in local mode using sample CSV files.

Run:

uvicorn service.app:app –reload

Open in browser:

http://127.0.0.1:8000/docs

------------------------------------------------------------------------

DATABASE MODE

To run with PostgreSQL (as in production / LMS checker):

export USE_DB=1 uvicorn service.app:app –reload

------------------------------------------------------------------------

API ENDPOINT

GET /post/recommendations/

Required parameters:
- id   : user id
- time : ISO datetime string (used to derive hour and day-of-week features)

Example request:

http://127.0.0.1:8000/post/recommendations/?id=51909&time=2026-02-16T12:00:00

Response example:

[ {“id”: 6676, “text”: “…”, “topic”: “movie”}, {“id”: 5978, “text”:
“…”, “topic”: “movie”}]

Returns exactly 5 recommendations.

------------------------------------------------------------------------

NOTES

-   Model weights are not included in the repository due to GitHub file size limits.
-   This repository focuses on inference and service architecture.
-   The system is designed to operate under realistic production constraints, 
    including latency and memory limits
