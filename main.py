from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-4B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torah_dtype="auto", device_map="auto")

msgs = [{"role": "user", "content": "What is 17 * 23?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(inputs, max_new_tokens=200)[0], skip_special_tokens=True))
