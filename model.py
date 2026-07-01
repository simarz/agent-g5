from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

msgs = [{"role": "user", "content": "What is 17 * 23?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
generated = outputs[0][inputs["input_ids"].shape[-1]:]
print(tok.decode(generated, skip_special_tokens=True))
