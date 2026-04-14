import boto3

# We check the model’s reliability before using it.
def validate_metrics(results_dict):
    return (
        results_dict.get("metrics/mAP50(B)", 0) >= 0.30
        and results_dict.get("metrics/recall(B)", 0) >= 0.25
    )

# Upload the two weight files to S3 after training:
def upload_model_to_s3(s3_client, best_path, last_path, bucket, run_name):
    s3_client.upload_file(best_path, bucket, f"models/{run_name}/best.pt")
    s3_client.upload_file(last_path, bucket, f"models/{run_name}/last.pt")
