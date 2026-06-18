from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from scraper import AlertWildfireScraper


# With Docker, the scraper.py file is located in the app/scraper folder
sys.path.insert(0, '/app/scraper')

# Scraping function import from the scraper.py file
def run_scraper():
    scraper = AlertWildfireScraper()
    scraper.start()
    try:
        scraper.scrape_all()
    finally:
        scraper.stop()

# 2 attempts in total before stopping
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# DAG to scrape the AlertWidlFire website every 5 minutes
# and save it to S3
with DAG(
    dag_id='fire_detection_scraper',
    default_args=default_args,
    description='Scrape AlertWildfire cameras every 5 minutes and upload to S3',
    schedule_interval='*/10 * * * *',
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    scrape_task = PythonOperator(
        task_id='scrape_cameras',
        python_callable=run_scraper,
    )
