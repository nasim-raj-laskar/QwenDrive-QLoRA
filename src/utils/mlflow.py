import os
import mlflow
import warnings
from dotenv import load_dotenv

load_dotenv()

def init_mlflow():
    """Initialize MLflow tracking with DagsHub token."""
    # Suppress MLflow warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
    
    # Set DagsHub credentials directly
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    
    # Configure MLflow with token authentication
    mlflow.set_tracking_uri(f"https://{dagshub_username}:{dagshub_token}@dagshub.com/nasim-raj-laskar/QwenDrive-QLoRA.mlflow")
    
    # Disable any interactive auth
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
    
    try:
        mlflow.set_experiment("QwenDrive-QLoRA")
    except Exception:
        pass  # Suppress any experiment creation errors