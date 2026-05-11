from datasets import load_dataset


def load_and_prepare(cfg, tokenizer):
    dataset = load_dataset("json", data_files=cfg["file"])
    split = dataset["train"].shuffle(seed=cfg["shuffle_seed"]).select(range(cfg["sample_size"]))

    system_prompt = cfg["system_prompt"]

    def format_chat(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["conversations"][0]["value"]},
            {"role": "assistant", "content": example["conversations"][1]["value"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    return split.map(format_chat)
