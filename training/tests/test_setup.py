import os
import boto3
import mlflow
from dotenv import load_dotenv



# Connexion test
load_dotenv()

def test_mlflow():
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    client.search_experiments()  # fails if server is unreachable

def test_s3():
    s3 = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"])
    s3.list_objects_v2(Bucket=os.environ["S3_BUCKET"], MaxKeys=1)
    # fails if bucket is unreachable or credentials are invalid

