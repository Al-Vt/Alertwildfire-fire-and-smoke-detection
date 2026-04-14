from dotenv import load_dotenv
from ultralytics import YOLO, settings
import mlflow
import pyfiglet
import os
import time
import argparse
import boto3

from utils import validate_metrics, upload_model_to_s3


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
    parser.add_argument("--weights", default="yolo11m.pt", type=str)
    parser.add_argument("--resume", action="store_true",
        help="Auto-charge last.pt depuis runs/detect/{run_name}/weights/last.pt")
    args = parser.parse_args()

    # Si --resume, on cherche automatiquement le dernier checkpoint
    if args.resume:
        auto_weights = f"runs/detect/{args.run_name}/weights/last.pt"
        if not os.path.exists(auto_weights):
            raise FileNotFoundError(f"Pas de checkpoint trouvé : {auto_weights}")
        args.weights = auto_weights
        print(f"[resume] Chargement depuis {args.weights}")

    # Setup MLflow
    EXPERIMENT_NAME = "fire-detection-01"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name) as run:

        model = YOLO(args.weights)

        mlflow.log_params({
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "freeze": args.freeze,
            "lr0": args.lr0,
            "hsv_v": args.hsv_v,
            "hsv_s": args.hsv_s,
            "run_name": args.run_name,
            "resumed_from": args.weights,
        })

        # Model training
        results = model.train(
            data=DATA_YAML,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            freeze=args.freeze,
            lr0=args.lr0,
            name="phase2",
            exist_ok=True,
            optimizer="Adam",
            # Data augmentation
            degrees=5,           # rotation
            scale=0.7,           # to spot fires from a distance or up close
            perspective=0.0005,
            hsv_v=args.hsv_v,   # brightness adjustment
            hsv_s=args.hsv_s,   # saturation
            flipud=0.0,          # no vertical symmetry, fire always rises upwards
        )

        mlflow.log_metrics({
            "mAP50": results.results_dict["metrics/mAP50(B)"],
            "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"],
        })
        mlflow.log_artifact(str(model.trainer.best))

        if not validate_metrics(results.results_dict):
            print("[warning] Le modèle n'a pas passé le quality gate (mAP50 < 0.30 ou recall < 0.25)")

        # Save model and checkpoint to S3
        s3 = boto3.client('s3', region_name=os.environ["AWS_DEFAULT_REGION"])
        upload_model_to_s3(
            s3,
            str(model.trainer.best),
            str(model.trainer.last),
            os.environ["S3_BUCKET"],
            args.run_name,
        )

    print(pyfiglet.figlet_format("training complete"))
    print(f"{args.run_name} best: {model.trainer.best}")
    print(f"Total training time: {time.time()-start_time:.1f}s")
