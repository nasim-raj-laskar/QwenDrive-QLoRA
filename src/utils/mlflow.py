import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

def init_mlflow():
    """Initialize MLflow tracking with DagsHub token."""
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_experiment("QwenDrive-QLoRA")