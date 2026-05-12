from datasets import load_dataset
from typing import Dict, List

def load_test_data(n_samples: int) -> List[Dict]:
    """Load and format test dataset."""
    dataset = load_dataset("json", data_files="data/automotive_en_dataset.jsonl")
    test_data = dataset["train"].shuffle(seed=123).select(range(n_samples))
    
    formatted_data = []
    for example in test_data:
        human = example["conversations"][0]["value"]
        assistant = example["conversations"][1]["value"]
        formatted_data.append({"input": human, "target": assistant})
    
    return formatted_data