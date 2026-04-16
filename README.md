# Fire Detection — Automated Wildfire Monitoring

ALERTWildfire.org operates 165 surveillance cameras across the western United States. Today, American volunteers manually watch these feeds to spot fires. This project replaces that manual process with a fine-tuned YOLO model that runs inference on camera snapshots every 5 minutes.

Built as a student project with a focus on production-grade MLOps practices: automated scraping, experiment tracking, drift monitoring, and a CI/CD pipeline.

---

## How it works

```
AlertWildfire Cameras
        ↓  scraped every 5 min (Airflow)
    S3 (raw images)
        ↓  YOLO inference
    Detections → Neon PostgreSQL
        ↓  drift check every hour (Evidently AI)
    Retraining triggered on EC2 if needed
```

The scraper grabs frames from all 165 cameras, runs them through the model, and stores detections with confidence scores. An hourly DAG checks whether the score distribution has drifted from the training baseline — if it has, it triggers a retraining job on EC2.

---

## Stack

| Layer | Tool |
|---|---|
| Detection model | YOLOv11 (fine-tuned) |
| Experiment tracking | MLflow |
| Orchestration | Apache Airflow |
| Storage | AWS S3 |
| Database | Neon PostgreSQL |
| Data warehouse | Snowflake |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |

---

## Project layout

```
├── training/
│   ├── train.py            # Training entry point (supports --resume)
│   ├── utils.py            # validate_metrics, upload_model_to_s3
│   └── tests/              # pytest unit tests
├── scraper/
│   ├── scraper.py          # Selenium scraper for AlertWildfire
│   └── database.py         # Neon DB interface
├── dags/
│   ├── scraper_dag.py              # ETL — every 5 min
│   ├── monitoring_dag.py           # Drift detection — hourly
│   └── continuous_training_dag.py  # SSH retraining on EC2
├── snowflake/
│   ├── schema.sql          # Star schema (FactDetection + dims)
│   └── load.py             # S3 → Snowflake loader
├── .github/workflows/
│   └── training.yml        # Run tests → build Docker image
└── docker-compose.yml
```

---

## CI/CD

Every push to `main` runs two jobs in sequence:

1. Unit tests (`pytest training/tests/`)
2. Docker build for the training image

---

## Running locally


To use this project, create a secrets.sh file with the following variables:
=======
**Start Airflow:**
>>>>>>> 2eb80c6 (Readme)

```bash
docker-compose up --build
```

Open `http://localhost:8080`.

**Train the model:**

```bash
cd training
python train.py --epochs 40 --freeze 10 --run_name V1
```

**Resume from a checkpoint:**

```bash
python train.py --resume --run_name V1 --epochs 20 --freeze 5 --lr0 0.001
```

---

## Environment variables

Create a `.env` at the project root (never commit it):

```bash
# AWS
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=
S3_BUCKET=

# MLflow
MLFLOW_TRACKING_URI=

# Database
DATABASE_URL=

# Snowflake
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_ACCOUNT=

# EC2
EC2_HOST=
EC2_KEY_PATH=
EC2_KNOWN_HOSTS=
```

---

## Dataset

The training set combines 6 public fire/smoke detection datasets into a single corpus of **109,121 images** (train: 85,627 / val: 12,451 / test: 11,043).

| # | Dataset | Source |
|---|---|---|
| 1 | Smoke-Fire-Detection-YOLO | Kaggle (sayedgamal99) |
| 2 | Fire/Smoke Detection YOLO v9 | Kaggle (roscoekerby) |
| 3 | fire-smoke-obstacle-dataset | Roboflow |
| 4 | D-Fire *(night only)* | Kaggle (shubhamkarande13) |
| 5 | FASDD — Flame And Smoke Detection *(night only)* | Public |
| 6 | Smoke night dataset *(night only)* | Roboflow |

ALERTWildfire cameras run 24/7, so the dataset was intentionally balanced toward night conditions: **51% of images are nighttime scenes**, filtered using brightness thresholds and CLIP-based relevance scoring.

---

## Model performance

| Metric | Value |
|---|---|
| mAP50 | 0.607 |
| Recall | 0.557 |

Training was stopped early due to cloud computing costs (student project). Similar research papers on fire detection with YOLO recommend training for ~500 epochs to reach convergence — we ran significantly fewer. The model hasn't fully converged, and recall in particular has room to improve. In fire detection, a missed fire is more dangerous than a false alarm, so recall is the metric that matters most.
