from datasets import load_dataset
from typing import Dict, List

def load_test_data(n_samples: int, config: Dict) -> List[Dict]:
    """Load and format test dataset."""
    dataset = load_dataset("json", data_files=config["data_file"])
    test_data = dataset["train"].shuffle(seed=config["test_seed"]).select(range(n_samples))
    
    formatted_data = []
    for example in test_data:
        human = example["conversations"][0]["value"]
        assistant = example["conversations"][1]["value"]
        formatted_data.append({"input": human, "target": assistant})
    
    return formatted_data