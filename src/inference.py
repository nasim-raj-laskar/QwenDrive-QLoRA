from transformers import pipeline


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 120):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

    result = pipe(
        text,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        return_full_text=False,
        eos_token_id=terminators,
    )
    return result[0]["generated_text"]
