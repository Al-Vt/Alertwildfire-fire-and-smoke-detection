from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import requests
import boto3
import logging

sys.path.insert(0, '/app/scraper')
from database import DatabaseManager

INFERENCE_API_URL = os.getenv("INFERENCE_API_URL")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")


def run_inference():
    db = DatabaseManager()
    s3 = boto3.client('s3')

    while True:
        images = db.get_pending_images(limit=10)
        if not images:
            break

        logging.info(f"image to processed : {len(images)}")

        for image in images:
            # Download the image from S3
            obj = s3.get_object(Bucket=S3_BUCKET, Key=image["s3_path"])
            image_bytes = obj["Body"].read()

            # Sends the image to the inference API, on HuggingFace Spaces
            response = requests.post(
                f"{INFERENCE_API_URL}/predict",
                files={"file": ("image.jpg", image_bytes)},
            )
            detections = response.json().get("confiance", [])

            # Detection 
            if detections:
                fire_detected = True
                confidence = detections[0]["confidence"]
                bbox = detections[0]["bbox"]
                logging.info(f"[ALERT] wildfire detected ! image_id={image['id']}, camera={image['camera_name']}")
            else:
                fire_detected = False
                confidence = 0.0
                bbox = []

            # In all cases, the result is written to Postgres.
            db.update_prediction(image["id"], fire_detected, confidence, bbox)

    db.close()



default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='fire_detection_inference',
    default_args=default_args,
    description='Run YOLO inference on pending images and update Postgres',
    schedule_interval='*/5 * * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    inference_task = PythonOperator(
        task_id='run_inference',
        python_callable=run_inference,
    )
