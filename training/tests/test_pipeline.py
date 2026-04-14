import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import validate_metrics, upload_model_to_s3

# Quality gate
def test_metrics_fail_low_recall():
    # We check that it correctly rejects the model if it misses too many fire cases.
    assert not validate_metrics({"metrics/mAP50(B)": 0.55, "metrics/recall(B)": 0.10})

# Upload test 
def test_s3_upload_correct_key():
    mock_s3 = MagicMock() # Creation of a mock S3 client 
    # Calling the function from utils.py
    upload_model_to_s3(mock_s3, "runs/V2/best.pt", "runs/V2/last.pt", "my-bucket", "V2") 
    # We verify that S3 received the correct arguments
    mock_s3.upload_file.assert_any_call("runs/V2/best.pt", "my-bucket", "models/V2/best.pt")
    mock_s3.upload_file.assert_any_call("runs/V2/last.pt", "my-bucket", "models/V2/last.pt")
