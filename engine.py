from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

def generate(prompt: str, *, max_new_tokens: int = 200, do_sample: bool = True,
             temperature: float = 0.8, top_p: float = 0.95) -> str:
    question = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(question, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    outputs = model.generate(**inputs, do_sample=do_sample, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tok.decode(generated, skip_special_tokens=True)
    return text


