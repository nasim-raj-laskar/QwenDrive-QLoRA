import os
import mlflow
import dagshub
from dotenv import load_dotenv
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

def init_mlflow():
    """Initialize MLflow tracking with DagsHub."""
    logger.info("Setting up MLflow with DagsHub...")    
    dagshub.auth.add_app_token(os.environ["DAGSHUB_TOKEN"])
    dagshub.init(
    repo_owner=os.environ["DAGSHUB_USERNAME"],
    repo_name=os.environ["DAGSHUB_REPO_NAME"],
    mlflow=True,
    )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("QwenDrive-QLoRA")
    logger.info(f"MLflow tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")