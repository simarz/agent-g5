from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random

model_id = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

test_data = load_dataset("openai/gsm8k", "main", split="test")
question_number = random(len(test_data))

msgs = [{"role": "user", "content": test_data[question_number]}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
generated = outputs[0][inputs["input_ids"].shape[-1]:]
print(tok.decode(generated, skip_special_tokens=True))
