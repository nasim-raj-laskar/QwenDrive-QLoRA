from transformers import pipeline


def run_inference(model, tokenizer, cfg: dict):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    messages = [{"role": "user", "content": cfg["prompt"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(cfg["eos_token"])]

    result = pipe(
        text,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        repetition_penalty=cfg["repetition_penalty"],
        do_sample=cfg["do_sample"],
        return_full_text=cfg["return_full_text"],
        eos_token_id=terminators,
    )
    return result[0]["generated_text"]
