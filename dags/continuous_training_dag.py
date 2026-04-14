from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import paramiko


def trigger_retraining():
    # reading environment variables
    ec2_host = os.environ["EC2_HOST"]
    ec2_key  = os.environ["EC2_KEY_PATH"]

    ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy()) # Reject unknown SSH key, for security
        ssh.load_host_keys(os.environ["EC2_KNOWN_HOSTS"])

    # Commmand in EC2
    ssh.exec_command(
        "cd /home/ubuntu/fire-detection && "
        "python train.py --resume --run_name auto_retrain --epochs 20" # training for 20 epochs
    )

    ssh.close()


with DAG(
    dag_id='fire_detection_continuous_training',
    description='Triggered when drift is detected — launches retraining on EC2',
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    retrain = PythonOperator(
        task_id='trigger_retraining',
        python_callable=trigger_retraining,
    )
