from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/app/scraper')
from scraper import AlertWildfireScraper

# Scraping function
def run_scraper():
    scraper = AlertWildfireScraper()
    scraper.start()
    try:
        scraper.scrape_all()
    finally:
        scraper.stop()


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='fire_detection_scraper',
    default_args=default_args,
    description='Scrape AlertWildfire cameras every 5 minutes and upload to S3',
    schedule_interval='*/5 * * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    scrape_task = PythonOperator(
        task_id='scrape_cameras',
        python_callable=run_scraper,
    )
