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
        
        # Tokenize and truncate to max length
        tokens = tokenizer(text, truncation=True, max_length=512, padding=False)
        
        return {
            "text": text,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }

    logger.info("Formatting dataset with chat template...")
    dataset = split.map(format_chat)
    
    # Filter out sequences that are too short or too long
    dataset = dataset.filter(lambda x: 10 <= len(x["input_ids"]) <= 512)
    logger.info(f"Filtered dataset to {len(dataset)} samples with valid lengths")

    if save_sample_path:
        with open(save_sample_path, "w") as f:
            for row in dataset:
                f.write(json.dumps({"text": row["text"]}) + "\n")
        logger.info(f"Saved {len(dataset)} sample rows to {save_sample_path}")

    logger.info("Dataset preparation completed")
    return dataset
