# Fire Detection 
## Automated Wildfire Monitoring

ALERTWildfire.org operates 165 surveillance cameras across the western United States. Today, American volunteers manually watch these feeds to spot fires. This project replaces that manual process with a fine-tuned YOLO model that runs inference on camera snapshots every 5 minutes.

Built as a student project with a focus on production-grade MLOps practices: automated scraping, experiment tracking, drift monitoring, and a CI/CD pipeline.

<img width="1095" height="840" alt="AlertWildFire" src="https://github.com/user-attachments/assets/3faa6753-9ebd-424a-a10b-394597154f3d" />
(View from the Alertwildfire.org website)                        

---

## Architecture

<img width="2040" height="1100" alt="Wildfire_pipeline" src="https://github.com/user-attachments/assets/0710902f-6b7d-452c-a606-b7a60f8ed452" />

---

## Dataset

The training set combines 6 public fire/smoke detection datasets into a single corpus of **109,121 images** (train: 85,627 / val: 12,451 / test: 11,043).

| # | Dataset | Source | Images |
|---|---|---|---|
| 1 | Smoke-Fire-Detection-YOLO | Kaggle (sayedgamal99) | ~21,000 |
| 2 | Fire/Smoke Detection YOLO v9 | Kaggle (roscoekerby) | ~28,000 |
| 3 | fire-smoke-obstacle-dataset | Roboflow | ~26,000 |
| 4 | D-Fire *(night only)* | Kaggle (shubhamkarande13) | ~17,000 |
| 5 | FASDD — Flame And Smoke Detection *(night only)* | Public | ~6,100 |
| 6 | Smoke night dataset *(night only)* | Roboflow | ~10,000 |
| | **Total** | | **109,121** |

ALERTWildfire cameras run 24/7, so the dataset was intentionally balanced toward night conditions: **51% of images are nighttime scenes**, filtered using brightness thresholds and CLIP-based relevance scoring.

---

## Stack

| Layer | Tool |
|---|---|
| Detection model | YOLOv11 (fine-tuned) |
| Experiment tracking | MLflow |
| Orchestration | Apache Airflow |
| Storage | AWS S3 |
| Database | Neon PostgreSQL |
| CI/CD | GitHub Actions |


---

## CI/CD

Every push to `main` runs two jobs in sequence:

1. Unit tests (`pytest training/tests/`)
2. Docker build for the training image

---

## Running locally

**Start Airflow:**

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

# EC2
EC2_HOST=
EC2_KEY_PATH=
EC2_KNOWN_HOSTS=
```

---

## Model performance

| Metric | Value |
|---|---|
| mAP50 | 0.607 |
| Recall | 0.557 |

Training was stopped early due to cloud computing costs. Similar research papers on fire detection with YOLO recommend training for ~500 epochs to reach convergence — we ran significantly fewer. The model hasn't fully converged, and recall in particular has room to improve. In fire detection, a missed fire is more dangerous than a false alarm, so recall is the metric that matters most.
