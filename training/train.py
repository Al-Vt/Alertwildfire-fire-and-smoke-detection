from dotenv import load_dotenv
from ultralytics import YOLO, settings
import mlflow
import pyfiglet 
import os
import time
import argparse
import boto3

if __name__ == "__main__":

    print(pyfiglet.figlet_format("training model"))

    # Time execution 
    start_time = time.time()

    # Load environment variables
    load_dotenv()
    DATA_YAML = os.environ["DATA_YAML"]
    MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]

    # YOLO has its own MLflow integration, we're disabling it
    settings.update({'mlflow': False}) 

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--freeze", default=10, type=int)
    parser.add_argument("--lr0", default=0.01, type=float)
    parser.add_argument("--hsv_v", default=0.5, type=float)
    parser.add_argument("--hsv_s", default=0.5, type=float)
    parser.add_argument("--run_name", default="V1", type=str)
    args = parser.parse_args()

    # Setup Mlflow
    EXPERIMENT_NAME = "fire-detection-01"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) 
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start Mlflow run
    with mlflow.start_run(run_name=args.run_name) as run: 

        model = YOLO("yolo11m.pt")

        mlflow.log_params({
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "freeze": args.freeze,
            "lr0": args.lr0,
            "hsv_v": args.hsv_v,
            "hsv_s": args.hsv_s,
            "run_name": args.run_name
        })

        # Model training
        results = model.train(
            data=DATA_YAML,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            freeze=args.freeze,
            lr0=args.lr0,
            name="phase1",
            exist_ok=True,
            optimizer="Adam",
            # Data augmentation
            degrees=5, # rotation
            scale=0.7, # to spot fires from a distance or up close 
            perspective=0.0005,
            hsv_v=args.hsv_v, # brightness adjustment.
            hsv_s=args.hsv_s, # Saturation
            flipud=0.0, # No vertical symmetry, fire always rises upwards
        )
        mlflow.log_metrics({
            "mAP50": results.results_dict["metrics/mAP50(B)"],
            "mAP50-95": results.results_dict["metrics/mAP50-95(B)"], # the strictest metric for defining bounding boxes 
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"], # We are trying to avoid false negatives  
        })
        mlflow.log_artifact(str(model.trainer.best))

        # Save model and checkpoint to S3 
        s3 = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"])
        s3.upload_file(
            str(model.trainer.best),
            os.environ["S3_BUCKET"],
            f"models/{args.run_name}/best.pt"
        )
        s3.upload_file(
            str(model.trainer.last),
            os.environ["S3_BUCKET"],
            f"models/{args.run_name}/last.pt"
        )

    print(pyfiglet.figlet_format("training complete"))
    print(f"{args.run_name} best: {model.trainer.best}")
    print(f"Total training time: {time.time()-start_time:.1f}s")

