import json
import hashlib
from pathlib import Path
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DatasetVersioner:
    def __init__(self, output_dir="data/versions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_version(self, dataset, config, preprocessing_steps):
        """Create a versioned snapshot of dataset metadata."""
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Compute dataset hash
        dataset_hash = self._compute_dataset_hash(dataset)
        
        # Create version metadata
        metadata = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "total_samples": len(dataset),
            "config": config,
            "preprocessing_steps": preprocessing_steps,
            "source_file": config.get("file"),
            "sample_size": config.get("sample_size"),
            "shuffle_seed": config.get("shuffle_seed"),
            "filters": {
                "min_tokens": config.get("min_seq_length", 10),
                "max_tokens": config.get("max_seq_length", 512)
            }
        }
        
        # Save metadata
        version_file = self.output_dir / f"version_{version_id}.json"
        with open(version_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created dataset version {version_id} with hash {dataset_hash[:8]}")
        return version_id, metadata
    
    def _compute_dataset_hash(self, dataset):
        """Compute hash of dataset content."""
        content = ""
        for example in dataset:
            prompt = example["conversations"][0]["value"]
            response = example["conversations"][1]["value"]
            content += f"{prompt}|||{response}\n"
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]