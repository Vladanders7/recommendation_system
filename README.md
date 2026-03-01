# Recommendation System (FastAPI + CatBoost)

![Language](https://badgen.net/badge/Language/Python%203.10%2B/blue)
![Framework](https://badgen.net/badge/Framework/FastAPI/009688)
![Model](https://badgen.net/badge/Model/CatBoost/yellow)
![Containerized](https://badgen.net/badge/Containerized/Docker/2496ED)
![Status](https://badgen.net/badge/Status/Completed/green)


End-to-end recommendation service that predicts **top-5 posts** a user is most likely to like.

---

## Architecture

### Training

- Raw feed logs  
- Feature engineering  
- Smoothed CTR computation (train only)  
- CatBoost training  
- Output: `model.cbm`

### Inference

- API request (`user_id`, `time`)
- Load precomputed user & post features
- Build feature matrix
- CatBoost `predict_proba`
- Return top-5 posts

**Production decision:**  
Post-level aggregated features (e.g., smoothed CTR) are precomputed and stored to avoid heavy recomputation during inference.

---

## Feature Engineering

### User features
- age
- city
- country
- exp_group
- gender
- os
- source

### Post features
- topic
- text_len
- smoothed CTR

### Context features
- hour
- day of week

### Smoothed CTR

CTR is computed using training data only (to prevent leakage):

CTR_smooth = (likes + alpha * global_ctr) / (views + alpha)

---

## Model

- **Model:** CatBoostClassifier  
- **Loss:** Logloss  
- **Evaluation metric:** AUC  
- Native categorical feature handling  

---

## Offline Evaluation (Leakage-Free)

Temporal split: **80% train / 20% validation**

**Validation metrics:**

- **AUC:** 0.643  
- **User-level HitRate@5:** 0.678  

**HitRate@5 definition:**

For each user in validation:
- Candidates = posts the user viewed in validation  
- If at least one liked post appears in top-5 → hit  

The model ranks at least one relevant post in top-5  
for ~68% of users with at least one like in validation.

---

## Online Evaluation (External API Checker)

The deployed service was evaluated in a production-like environment  
via an external API checker (2000 requests).

**Result:**  
- **HitRate@5 ≈ 0.60**

Note: This metric was computed using the checker’s internal evaluation protocol  
and differs from the offline validation described above.

---

## How to Run


### Run with Docker

Build image:

```bash
docker build -t recsys .
```

Run container:

```bash
docker run --rm -p 8000:8000 recsys
```

Service will be available at:
```
http://localhost:8000/docs
```

### Example request

```
curl "http://localhost:8000/post/recommendations/?id=51909&time=2026-02-16T12:00:00"
```

Example response:
```
[
  {"id": 6676, "text": "...", "topic": "movie"},
  {"id": 5978, "text": "...", "topic": "movie"},
  {"id": 1234, "text": "...", "topic": "sport"},
  {"id": 9981, "text": "...", "topic": "tech"},
  {"id": 4455, "text": "...", "topic": "news"}
]
```
Returns exactly 5 recommendations

---

## Repository Structure

```
recommendation_system/
├── src/                  # training & evaluation scripts
├── data/sample/          # sample data for local run
├── artifacts/            # trained model (local)
├── app.py                # FastAPI service entrypoint
├── requirements.txt
└── README.md
```

---

## Key Engineering Decisions

- Temporal split to prevent leakage
- CTR computed on train only  
- Clear separation of offline vs online evaluation  
- Precomputed aggregated features for fast inference 

---

## Project Goal

Demonstrate ability to:
- Build an end-to-end ML pipeline
- Implement leakage-free validation
- Deploy model via API
- Distinguish offline vs online metrics
- Design scalable inference logic
