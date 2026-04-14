import os 
import mlflow 
from dotenv import load_dotenv

load_dotenv()

def test_mlflow_experiment_creation():
    # Checks that we can create an MLflow experiment and that it appears in S3
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"]) # HuggingFace server
    mlflow.set_experiment("test-integration")
    
    # Opens a real MLflow run on the server 
    with mlflow.start_run(run_name="test") as run:
        #"Log dummy params and metrics"
        mlflow.log_param("test_param", 42)
        mlflow.log_metric("test_metric", 0.99)
    
    # Checks that the run exists
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["1"])
    assert len(runs) > 0 # Checks that at least one run exists 
    # fails if MLflow can't log to the server
