#test fused model
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-MLXTuned")

prompt = "What is under-fitting and overfitting in machine learning?"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
