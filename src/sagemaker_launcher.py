import os
import yaml
from datetime import datetime
from sagemaker.huggingface import HuggingFace


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def launch_sagemaker_job():
    sm_cfg = load_config("configs/sagemaker.yaml")

    huggingface_estimator = HuggingFace(
        entry_point="scripts/train_sm.py",
        source_dir=".",
        role=os.getenv("SAGEMAKER_ROLE_ARN"),
        instance_type=sm_cfg["instance_type"],
        instance_count=sm_cfg["instance_count"],
        volume_size=sm_cfg["volume_size_gb"],
        max_run=sm_cfg["max_runtime_seconds"],
        transformers_version=sm_cfg["transformers_version"],
        pytorch_version=sm_cfg["pytorch_version"],
        py_version=sm_cfg["python_version"],
        output_path=sm_cfg["s3_output_uri"],
        base_job_name=sm_cfg["job_name_prefix"],
        environment={
            "TRANSFORMERS_CACHE": "/tmp/transformers_cache",
            "HF_HOME": "/tmp/hf_home",
        },
    )

    print(f"Launching SageMaker job on {sm_cfg['instance_type']}...")
    print(f"Data:   {sm_cfg['s3_data_uri']}")
    print(f"Output: {sm_cfg['s3_output_uri']}")

    huggingface_estimator.fit({"training": sm_cfg["s3_data_uri"]}, wait=True)

    print(f"\nDone. Model artifacts at: {huggingface_estimator.model_data}")
