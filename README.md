# Fire Detection — Automated Wildfire Monitoring System

## Overview

This project implements an automated wildfire detection system using computer vision to analyze real-time video feeds from 165 surveillance cameras hosted on [ALERTWildfire.org](https://www.alertwildfire.org).

ALERTWildfire.org currently relies on American citizens to manually monitor camera feeds. This project automates that process using a fine-tuned YOLO model, enabling faster and more reliable fire detection at scale.

---

## Architecture

```
AlertWildfire Cameras
        ↓ (Airflow — every 5 min)
    S3 (raw images)
        ↓ (YOLO inference)
    Detections (Neon PostgreSQL)
        ↓ (Evidently AI — hourly)
    Drift Monitoring
        ↓ (if drift detected)
    Retraining on EC2
```

---

## Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv11 (fine-tuned) |
| Experiment Tracking | MLflow |
| Orchestration | Apache Airflow |
| Storage | AWS S3 |
| Database | Neon PostgreSQL |
| Data Warehouse | Snowflake |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |

---

## Project Structure

```
├── training/           # Model training pipeline
│   ├── train.py        # Training script
│   ├── utils.py        # Reusable functions
│   └── tests/          # Unit tests
├── scraper/            # Camera scraping pipeline
│   ├── scraper.py      # AlertWildfire scraper
│   └── database.py     # Neon DB manager
├── dags/               # Airflow DAGs
│   ├── scraper_dag.py              # ETL — scrape every 5 min
│   ├── monitoring_dag.py           # Drift detection — hourly
│   └── continuous_training_dag.py  # Retraining on EC2
├── snowflake/          # Data warehouse
│   ├── schema.sql      # Star schema (FactDetection)
│   └── load.py         # S3 → Snowflake loader
├── .github/workflows/  # CI/CD
│   └── training.yml    # Run tests + build Docker on push
└── docker-compose.yml  # Airflow + Postgres
```

---

## CI/CD Pipeline

On every push to `main`, GitHub Actions automatically:
1. Runs unit tests (`pytest training/tests/`)
2. Builds the training Docker image

---

## Environment Variables

Create a `.env` file at the root (never commit it):

```bash
# AWS
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
S3_BUCKET=...

# MLflow
MLFLOW_TRACKING_URI=...

# Database
DATABASE_URL=...

# Snowflake
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_ACCOUNT=...

# EC2
EC2_HOST=...
EC2_KEY_PATH=...
EC2_KNOWN_HOSTS=...
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| mAP50  | 0.607 |
| Recall | 0.557 |

> **Note:** This is a student project with limited budget. Training was stopped early due to cloud computing costs. The model has not fully converged and can be further trained to improve performance — particularly recall, which is the most critical metric for fire detection (a missed fire is more dangerous than a false alarm).

---

## Getting Started

**Run Airflow locally:**
```bash
docker-compose up --build
```
Then open `http://localhost:8080`.

**Run training:**
```bash
cd training
python train.py --epochs 40 --freeze 10 --run_name V1
```

**Resume training:**
```bash
python train.py --resume --run_name V1 --epochs 20 --freeze 5 --lr0 0.001
```
