def create_zero_shot_prompt(text):
    text = text[:200] if len(text) > 200 else text
    return f"""Classification task: Determine if the text is written in formal or informal language.

Text: "{text}"

Answer with ONLY ONE WORD - either "formal" or "informal":"""

def create_few_shot_prompt(text, examples):
    text = text[:200] if len(text) > 200 else text
    prompt = "Task: Determine if the following texts are formal or informal.\n\n"
    for ex_text, ex_label in examples:
        ex_text = ex_text[:200] if len(ex_text) > 200 else ex_text
        prompt += f"Text: {ex_text}\nAnswer: {ex_label}\n\n"
    prompt += f"Text: {text}\nAnswer:"
    return prompt
