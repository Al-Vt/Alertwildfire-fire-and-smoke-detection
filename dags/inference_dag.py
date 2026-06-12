from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import requests
import boto3
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from PIL import Image, ImageDraw
import io


sys.path.insert(0, '/app/scraper')
from database import DatabaseManager

INFERENCE_API_URL = os.getenv("INFERENCE_API_URL")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")


# Alert if wildfire detected
def send_alert_email(camera_name, confidence, image_bytes, bbox):
    # Take the image returned by the model
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # and draw the bounding box
    ImageDraw.Draw(image).rectangle(bbox, outline="red", width=5)

    # converting the image to JPEG
    output = io.BytesIO()
    image.save(output, format="JPEG")

    # building email
    msg = MIMEMultipart()
    msg["Subject"] = f"[ALERTE WILDFIRE] Camera {camera_name}"
    msg["From"] = os.getenv("ALERT_EMAIL_SENDER")
    msg["To"] = os.getenv("ALERT_EMAIL_RECEIVER")
    # Texte with camera name and confidence score 
    msg.attach(MIMEText(f"Fire detected on camera {camera_name} (confidence : {confidence:.2f})"))
    # image returned by the model
    msg.attach(MIMEImage(output.getvalue(), name=f"{camera_name}.jpg"))

    # sent via Gmail
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login(os.getenv("ALERT_EMAIL_SENDER"), os.getenv("ALERT_EMAIL_PASSWORD"))
    server.sendmail(msg["From"], msg["To"], msg.as_string())
    server.quit()


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
            detections = response.json().get("detections", [])

            # Detection 
            if detections:
                fire_detected = True
                # The first box you choose is the one with the highest level of trust
                confidence = detections[0]["confidence"]
                bbox = detections[0]["bbox"]
                logging.info(f"[ALERT] wildfire detected, image_id={image['id']}, camera={image['camera_name']}")
                send_alert_email(image["camera_name"], confidence, image_bytes, bbox)
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
