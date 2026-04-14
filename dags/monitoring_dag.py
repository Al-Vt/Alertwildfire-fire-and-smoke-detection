from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime
import sys

# Adding a directory to import code
sys.path.insert(0, '/app/scraper')
from database import DatabaseManager

REFERENCE_CONFIDENCE = 0.70
DRIFT_THRESHOLD = 0.10

# Detection of data drift in real-world data
def detect_drift():
    db = DatabaseManager()  # Allows access to stored data
    images = db.get_pending_images(limit=100)
    db.close() # Prevents memory leaks and unnecessary open connections

    # list of confidence scores
    confidences = [img['confidence'] for img in images if img.get('confidence')]
    if not confidences: # si aucune donnée n'est valide
        return 'no_drift_detected'

    # calculation of the average confidence
    avg = sum(confidences) / len(confidences)
    if abs(avg - REFERENCE_CONFIDENCE) > DRIFT_THRESHOLD:
        return 'data_drift_detected'
    return 'no_drift_detected'

# concluding message
def log_drift():
    print("[ALERT] Drift détecté — penser à ré-entraîner le modèle")

def log_no_drift():
    print("[OK] Pas de drift")

with DAG(
    dag_id='fire_detection_monitoring',
    schedule_interval='@hourly',
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    branch = BranchPythonOperator(task_id='detect_drift', python_callable=detect_drift)
    drift = PythonOperator(task_id='data_drift_detected', python_callable=log_drift)
    no_drift = PythonOperator(task_id='no_drift_detected', python_callable=log_no_drift)

    branch >> [drift, no_drift]
