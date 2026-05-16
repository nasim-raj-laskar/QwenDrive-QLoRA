import json
from datasets import load_dataset
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_and_prepare(cfg, tokenizer, save_sample_path=None):
    logger.info(f"Loading dataset from {cfg['file']}")
    dataset = load_dataset("json", data_files=cfg["file"])
    split = dataset["train"].shuffle(seed=cfg["shuffle_seed"]).select(range(cfg["sample_size"]))
    logger.info(f"Selected {cfg['sample_size']} samples from dataset")

    system_prompt = cfg["system_prompt"]
    logger.info(f"Using system prompt: {system_prompt}")

    def format_chat(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["conversations"][0]["value"]},
            {"role": "assistant", "content": example["conversations"][1]["value"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Tokenize and truncate to max length from config
        max_length = cfg.get("max_seq_length", 512)
        tokens = tokenizer(text, truncation=True, max_length=max_length, padding=False)
        
        return {
            "text": text,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }

    logger.info("Formatting dataset with chat template...")
    formatted_data = split.map(format_chat)
    
    # Filter out sequences that are too short or too long using config values
    min_length = cfg.get("min_seq_length", 10)
    max_length = cfg.get("max_seq_length", 512)
    filtered_data = formatted_data.filter(lambda x: min_length <= len(x["input_ids"]) <= max_length)
    logger.info(f"Filtered dataset to {len(filtered_data)} samples with valid lengths")

    # Split into train/val/test using config ratios
    val_test_ratio = cfg["splits"]["validation"] + cfg["splits"]["test"]
    splits = filtered_data.train_test_split(
        test_size=val_test_ratio,
        seed=cfg["shuffle_seed"]
    )
    
    # Further split the test portion into val and test
    test_ratio = cfg["splits"]["test"] / val_test_ratio
    val_test_splits = splits["test"].train_test_split(
        test_size=test_ratio,
        seed=cfg["shuffle_seed"]
    )
    
    train_dataset = splits["train"]  
    val_dataset = val_test_splits["train"]  
    test_dataset = val_test_splits["test"]  
    
    logger.info(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    if save_sample_path:
        with open(save_sample_path, "w") as f:
            for row in train_dataset:
                f.write(json.dumps({"text": row["text"]}) + "\n")
        logger.info(f"Saved {len(train_dataset)} training sample rows to {save_sample_path}")

    logger.info("Dataset preparation completed")
    return {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }
