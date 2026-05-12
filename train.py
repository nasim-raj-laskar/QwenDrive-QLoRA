from dotenv import load_dotenv
from src.sagemaker_launcher import launch_sagemaker_job

if __name__ == "__main__":
    load_dotenv()
    launch_sagemaker_job()
